#!/usr/bin/env python3
# PIFOAMUI - a browser front end for OpenFOAM 13
# Copyright (C) 2026 BiscuitIV
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details: <https://www.gnu.org/licenses/>.
#
# Several features are ported from Magnusim (github.com/Lilmill2000/Magnusim, GPL-3.0-or-later).
# The embedded IBM Plex fonts are under the SIL Open Font License 1.1 (see the note above _FONT_FILES).
# Not affiliated with or endorsed by OpenCFD Ltd, ESI or the OpenFOAM Foundation.
# OPENFOAM(R) is a registered trademark of OpenCFD Ltd.
"""
PIFOAMUI.py - a browser front end for OpenFOAM 13 (external flow around an STL).

Run inside Linux or WSL:   python3 PIFOAMUI.py
Then open in your browser:   http://localhost:8080
No extra packages needed (Python 3 standard library only).

Environment variables (all optional):
  PIFOAM_BASHRC    OpenFOAM 13's etc/bashrc     (default /opt/openfoam13/etc/bashrc)
  PIFOAM_TUTORIAL  the motorBike tutorial case  (default $FOAM_TUTORIALS/incompressibleFluid/motorBike/motorBike)
  PIFOAM_PORT      HTTP port                    (default 8080; the server only listens on 127.0.0.1)

Includes a case manager (list/switch/delete cases under $FOAM_RUN), an STL
preview + live convergence chart (hand-rolled <canvas>, no CDN), automated
mesh-quality retry, solver auto-stop on convergence, parameter-sweep batch
runs, a FIFO job queue, parallel (MPI) meshing/solving, a fluid material
library, boundary layers + wake refinement, inlet/outlet monitors, a residual
plot, preset ParaView views, Basic/Advanced modes and "use case as template".
"""
import csv
import glob
import html
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from string import Template

PORT = int(os.environ.get("PIFOAM_PORT", 8080))
BASHRC = os.environ.get("PIFOAM_BASHRC", "/opt/openfoam13/etc/bashrc")


def foam_bash(cmd):
    """argv running cmd in bash with the OpenFOAM environment sourced."""
    return ["bash", "-c", f"source {shlex.quote(BASHRC)}; {cmd}"]


def sh(cmd):
    r = subprocess.run(foam_bash(cmd), capture_output=True, text=True)
    return r.stdout.strip()


RUN = sh("echo $FOAM_RUN") or os.path.expanduser("~/OpenFOAM/run")
# the motorBike tutorial is the case template (see setup_case)
TUTORIAL = os.environ.get("PIFOAM_TUTORIAL") or os.path.join(
    sh("echo $FOAM_TUTORIALS") or "/opt/openfoam13/tutorials",
    "incompressibleFluid/motorBike/motorBike")
STATE = {"verts": None, "name": ""}
JOB = {"proc": None, "running": False, "stop": False,
       "kind": None, "case": None, "base": None, "meta": {}, "last_finish": None, "seq": 0}
LOG = []          # list of text chunks
LOG_LEN = [0]
LOG_LOCK = threading.Lock()
LAST_SWEEP = {}   # last completed sweep: {"param": ..., "results": [...]}
QUEUE = []        # pending jobs: {"id", "kind", "case", "fn", "args", "kwargs"}
QUEUE_LOCK = threading.Lock()
QUEUE_STATE = {"worker": False, "next_id": 0}
BOOT = time.time()   # identifies this server process: the browser reloads once it changes after a restart


def physical_cores():
    """Physical cores (unique physical id/core id pairs in /proc/cpuinfo): MPI
    CFD runs best at one rank per physical core - hyperthreads mostly slow it."""
    try:
        pairs, phys = set(), "0"
        for line in open("/proc/cpuinfo"):
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k == "physical id":
                phys = v
            elif k == "core id":
                pairs.add((phys, v))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


THREADS = os.cpu_count() or 1
CORES = physical_cores()
DEFAULT_NPROCS = CORES

# fluid presets at 20 C, 1 atm: key -> (label, kinematic viscosity nu [m2/s], density rho [kg/m3])
MATERIALS = {
    "air": ("Air", 1.5e-5, 1.204),
    "water": ("Water", 1.004e-6, 998.2),
    "seawater": ("Seawater (3.5% saline)", 1.05e-6, 1025.0),
    "co2": ("Carbon dioxide", 8.03e-6, 1.842),
    "argon": ("Argon", 1.34e-5, 1.661),
    "hydrogen": ("Hydrogen", 1.05e-4, 0.0838),
    "nitrogen": ("Nitrogen", 1.5e-5, 1.165),
}

# simulation types a new case can be created as (key -> label); stored in gui.json params["sim_type"]
SIM_TYPES = {"external": "External flow (steady)", "transient": "External flow (transient)"}   # incompressible

# mesh resolution ('mesh_preset'). The form always shows a background and a surface cell size in mm
# ("surface" mode, labelled Custom); a preset just fills them: key -> (label, surface cell mm, background
# cell budget). The background cell is sized so the domain holds about that many cells, and the refinement
# level is the fewest halvings from it down to the surface size.
MESH_PRESETS = {"surf4": ("Coarse", 4, 150000), "surf2": ("Medium", 2, 250000),
                "surf1": ("Fine", 1, 400000), "surf05": ("Very fine", 0.5, 600000)}
# Older cases saved other modes, still computed for them (ParaView, report) and converted to "surface" by
# case_template() on load: these presets (key -> (part length / background cell, level)), "custom"
# ('cell_size_mm' + 'level'), "count" ('target_cells' + 'level'), 'cells_per_part' or a raw 'cells' count.
LEGACY_PRESETS = {"coarse": (4, 2), "medium": (6, 3), "fine": (9, 3)}
MAX_LEVEL = 10   # highest surface refinement level a resolution may need

# parameters a sweep may vary: name -> value parser (raises ValueError on a bad value)
PRESET_ALIASES = {"coarse": "surf4", "medium": "surf2", "fine": "surf1", "very fine": "surf05", "vfine": "surf05"}
def _preset(v):
    v = PRESET_ALIASES.get(v.strip().lower(), v.strip())
    if v not in MESH_PRESETS:
        raise ValueError("Mesh resolution must be one of coarse, medium, fine, very fine.")
    return v


SWEEP_PARAMS = {"speed": float, "dir": str, "aoa": float, "yaw": float,
                "mesh_preset": _preset, "cell_size_mm": float, "target_cells": float, "surface_cell_mm": float}
GROUND_TYPES = ("none", "fixed", "moving")
# half model: off; the full part cut by a symmetry plane across the span axis at its bbox centre or at
# half_mm (the + side is meshed); or (older cases) a part that is already one half, root at its span min / max
HALF_MODELS = ("none", "centre", "custom", "min", "max")
HALF_INSET = 1e-3       # an already-half part's symmetry plane sits this fraction of the span inside its root
# RAS models (key -> label); the key is written to constant/momentumTransport
TURB_MODELS = {"SpalartAllmaras": "Spalart–Allmaras", "kOmegaSST": "k-ω SST",
               "kOmegaSSTLM": "k-ω SST with γ-Reθ transition"}
MAX_ANGLE = 30          # |angle of attack| and |yaw| limit (deg); larger tilts leave the axis-aligned domain
# features a case can switch off (key -> (label, the params they reset to neutral)); a case's 'features' param
# lists the disabled keys (missing = all on). Injected into PAGE via /*FEATURE_GROUPS*/null; apply_features enforces
FEATURE_GROUPS = {
    "aircraft": ("Aircraft analysis", {"ref_mode": "auto", "mass": "", "target_sm": ""}),
    "angles": ("Angle of attack / yaw", {"aoa": "0", "yaw": "0"}),
    "ground": ("Wind-tunnel ground", {"ground": "none"}),
    "half": ("Half model", {"half_model": "none"}),
    "probes": ("Probes & sample lines", {"probe_points": "", "probe_lines": ""}),
    "studies": ("Studies (sweeps)", {}),
    "compare": ("Compare cases", {}),
}
AVG_WINDOW = 100        # default number of final iterations coefficients are averaged over
G = 9.81                # gravity (m/s2), for flight figures from the aircraft mass


def log(text):
    with LOG_LOCK:
        LOG.append(text)
        LOG_LEN[0] += len(text)


# ---------------------------------------------------------------- STL helpers
def read_stl(data):
    """Return list of (x, y, z) vertices, 3 per triangle. Handles ASCII and binary STL."""
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            verts = []
            for i in range(n):
                off = 84 + 50 * i + 12
                for k in range(3):
                    verts.append(struct.unpack("<3f", data[off + 12 * k: off + 12 * k + 12]))
            return verts
    nums = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)   # on the bytes: no decode of a 100 MB file
    return [(float(a), float(b), float(c)) for a, b, c in nums]


def write_ascii_stl(path, verts, name="part"):
    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for i in range(0, len(verts) - 2, 3):
            a, b, c = verts[i], verts[i + 1], verts[i + 2]
            u = [b[j] - a[j] for j in range(3)]
            v = [c[j] - a[j] for j in range(3)]
            n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
            m = math.sqrt(sum(x * x for x in n)) or 1.0
            f.write(f"facet normal {n[0]/m:.6e} {n[1]/m:.6e} {n[2]/m:.6e}\nouter loop\n")
            for p in (a, b, c):
                f.write(f"vertex {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n")
            f.write("endloop\nendfacet\n")
        f.write(f"endsolid {name}\n")


def bbox(verts):
    mn = [min(v[i] for v in verts) for i in range(3)]
    mx = [max(v[i] for v in verts) for i in range(3)]
    return mn, mx


PREVIEW_TRIS = 12000   # triangles in the browser's part preview


def stl_preview(verts, max_tris=PREVIEW_TRIS):
    """Simplified copy of the surface for the browser's <canvas> preview -
    independent from the full-resolution verts used for meshing, to keep the
    payload small. Vertex clustering: vertices are snapped to the mean of their
    grid cell, collapsed and duplicate triangles are dropped, so the preview
    stays a closed surface (dropping every Nth triangle would leave holes).
    The grid resolution is steered toward max_tris (the triangle count grows ~ res^2), keeping the finest
    result that fits: a fixed guess would leave thin parts such as wings with only a few hundred triangles."""
    n_tris = len(verts) // 3
    if n_tris <= max_tris:
        return [tuple(round(c, 6) for c in v) for v in verts]
    mn, mx = bbox(verts)
    size = max(mx[i] - mn[i] for i in range(3)) or 1.0
    res, best = max(8, int((max_tris / 2) ** 0.5)), None   # a bulky closed surface: ~2-6 tris per res^2
    for _ in range(6):
        out = _cluster(verts, mn, size / res)
        n = len(out) // 3
        if n <= max_tris and (best is None or n > len(best) // 3):
            best = out
        if 0.6 * max_tris <= n <= max_tris or (n > max_tris and res <= 8):
            break
        res = max(8, int(res * min(3.0, 0.95 * (max_tris / max(n, 1)) ** 0.5)))
    return best if best is not None else out


def _cluster(verts, mn, cell):
    """stl_preview's vertex clustering at one grid cell size."""
    keys = [(int((v[0] - mn[0]) / cell), int((v[1] - mn[1]) / cell), int((v[2] - mn[2]) / cell)) for v in verts]
    acc = {}
    for k, v in zip(keys, verts):
        a = acc.setdefault(k, [0.0, 0.0, 0.0, 0])
        a[0] += v[0]; a[1] += v[1]; a[2] += v[2]; a[3] += 1
    rep = {k: (round(a[0] / a[3], 6), round(a[1] / a[3], 6), round(a[2] / a[3], 6)) for k, a in acc.items()}
    seen, out = set(), []
    for i in range(0, len(keys) - 2, 3):
        tri = keys[i:i + 3]
        if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
            continue
        s = tuple(sorted(tri))
        if s in seen:
            continue
        seen.add(s)
        out.extend(rep[k] for k in tri)
    return out


# ---------------------------------------------------------------- templates
FOAMFILE = "FoamFile { format ascii; class %s; object %s; }\n"

BLOCKMESH = Template("""FoamFile { format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
    ($x0 $y0 $z0) ($x1 $y0 $z0) ($x1 $y1 $z0) ($x0 $y1 $z0)
    ($x0 $y0 $z1) ($x1 $y0 $z1) ($x1 $y1 $z1) ($x0 $y1 $z1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ($nx $ny $nz) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet  { type patch; faces ($inlet); }
    outlet { type patch; faces ($outlet); }
    sides  { type $sidetype; faces ($sides); }
    $extrapatches
);
""")

FEATURES = Template("""FoamFile { format ascii; class dictionary; object surfaceFeaturesDict; }
surfaces ("part.stl");
includedAngle $angle;
writeObj no;
""")

SNAPPY = Template("""FoamFile { format ascii; class dictionary; object snappyHexMeshDict; }
castellatedMesh true;
snap            true;
addLayers       $addlayers;
geometry
{
    body { type triSurface; file "part.stl"; }
    $wake_geom
}
castellatedMeshControls
{
    maxLocalCells $maxcells;
    maxGlobalCells $maxcells;
    minRefinementCells 0;
    nCellsBetweenLevels $nbetween;
    features ( { file "part.eMesh"; level $level; } );
    refinementSurfaces { body { level ($level $level); } }
    resolveFeatureAngle $resolve;
    refinementRegions { $wake_region }
    locationInMesh ($lx $ly $lz);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    layers { $layers }
    $thickness
    nGrow 0;
    featureAngle 130;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}
meshQualityControls
{
    #includeEtc "caseDicts/mesh/generation/meshQualityDict"
}
mergeTolerance 1e-6;
""")

CONTROL = Template("""FoamFile { format ascii; class dictionary; object controlDict; }
application     foamRun;
solver          incompressibleFluid;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         $endtime;
$timecontrol
purgeWrite      0;
writeFormat     binary;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
    forces
    {
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (body);
        rho             rhoInf;
        rhoInf          $rho;
        CofR            ($cx $cy $cz);
    }
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (body);
        rho             rhoInf;
        rhoInf          $rho;
        CofR            ($cx $cy $cz);
        liftDir         ($lift);
        dragDir         ($drag);
        pitchAxis       ($pitch);
        magUInf         $speed;
        lRef            $lref;
        Aref            $aref;
    }
    residuals
    {
        type            residuals;
        libs            ("libutilityFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          ($resfields);
    }
    inletAvg
    {
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        patch           inlet;
        operation       areaAverage;
        fields          (p U);
    }
    outletFlow
    {
        type            surfaceFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        writeFields     false;
        patch           outlet;
        operation       sum;
        fields          (phi);
    }
$finalfields$probes}
""")

# controlDict time control: a steady run counts iterations (time step 1); a transient one runs in seconds
# with the time step adjusted to the Courant limit, writing on exact multiples of the write interval
STEADY_TIME = Template("""deltaT          1;
writeControl    timeStep;
writeInterval   $writeinterval;""")
TRANSIENT_TIME = Template("""deltaT          $deltat;
writeControl    adjustableRunTime;
writeInterval   $writeinterval;
adjustTimeStep  yes;
maxCo           $maxco;
maxDeltaT       $writeinterval;""")
FINAL_FIELDS = Template("""    finalFields
    {
        // OpenFOAM writes fields only every writeInterval iterations, so a run ending on endTime
        // in between (e.g. 50 iterations, interval 100) would leave nothing for ParaView;
        // solves always start from 0, so iteration endTime is time step endTime
        type            writeObjects;
        libs            ("libutilityFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   $endtime;
        objects         ($writefields);
    }
""")

PHYSICAL = FOAMFILE % ("dictionary", "physicalProperties") + \
    "viscosityModel  constant;\nnu              [0 2 -1 0 0 0 0] %g;\n"

DECOMPOSE = FOAMFILE % ("dictionary", "decomposeParDict") + \
    "numberOfSubdomains %d;\ndecomposer      scotch;\ndistributor     ptscotch;\n"

# Predefined ParaView views (value -> button label/description is the PAGE's JS 'PV' list)
PV_VIEWS = ("overview", "uslice", "pslice", "streamlines", "bodyp", "wake")

# pvpython script for the ParaView button (run from the case dir), prefixed by
# the server with a 'CFG = {...}' line: the chosen view plus compute_geometry()
# axes/extents (no geometry keys -> overview only). ParaView's reader skips
# the 0/ dir by default, which leaves a mesh-only case with no fields at all;
# a parallel run still in progress only has results in processor*/.
PARAVIEW_SCRIPT = """import os
from paraview.simple import *

def times(d):
    out = []
    for n in os.listdir(d) if os.path.isdir(d) else []:
        try:
            if float(n) > 0:
                out.append(n)
        except ValueError:
            pass
    return out

DECOMPOSED = os.path.isdir('processor0') and not times('.') and times('processor0')

def reader(want_body):
    # one reader per region (internal mesh / body patch): avoids ExtractBlock,
    # whose selector syntax differs between ParaView versions
    r = OpenFOAMReader(FileName='case.foam')
    r.SkipZeroTime = 0
    try:
        # keep snappy's split-hex polyhedra whole: decomposing them adds a point at each cell centre,
        # and slices through the resulting tet fans show zigzag colour artefacts
        r.Decomposepolyhedra = 0
    except AttributeError:
        pass
    if DECOMPOSED:
        r.CaseType = 'Decomposed Case'
    r.UpdatePipelineInformation()
    if want_body:
        try:
            avail = list(r.MeshRegions.Available)
        except Exception:
            avail = ['patch/body', 'body']
        found = [n for n in avail if n == 'body' or n.endswith('/body')]
        if not found:
            print("PIFOAM: the mesh has no 'body' patch, so the part isn't in it. Mesh the case (and solve it) first.")
        r.MeshRegions = found or ['patch/body', 'body']
    else:
        r.MeshRegions = ['internalMesh']
    return r

view = GetActiveViewOrCreate('RenderView')
try:
    view.UseFXAA = 1   # anti-aliasing: smooths jagged slice and part edges
except AttributeError:
    pass
internal, body = reader(False), reader(True)

def reflect(src):
    # a half model: the solved half plus its mirror image across the symmetry plane, so every view shows the
    # whole part. Flip every vector array (U), not just the active one, or the mirrored flow crosses the plane
    r = Reflect(Input=src)
    r.Plane = 'XYZ'[CFG['sym'][0]]
    r.Center = CFG['sym'][1]
    r.CopyInput = 1
    try:
        r.FlipAllInputArrays = 1
    except AttributeError:
        pass
    return r

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
tv = internal.TimestepValues   # a bare float when there is only one time step (a reader's property: before Reflect)
tv = list(tv) if hasattr(tv, '__len__') else [tv]
if tv:
    scene.AnimationTime = tv[-1]
if CFG.get('sym'):
    internal, body = reflect(internal), reflect(body)

V = CFG.get('view', 'overview') if 'ctr' in CFG else 'overview'

def unit(i):
    return [1.0 if k == i else 0.0 for k in range(3)]

SPEED = CFG.get('speed')

def color(disp, field, title, presets, comp=None, fmt='%-#.2f'):
    # one colour path for every view: named preset, range fitted to the shown data at the
    # displayed time (so both ends of the map appear), no auto-rescale when the time
    # changes, and a titled scalar bar with units
    ColorBy(disp, ('POINTS', field, comp) if comp else ('POINTS', field))
    lut = GetColorTransferFunction(field)
    for name in presets:
        if lut.ApplyPreset(name, True):
            break
    lut.AutomaticRescaleRangeMode = 'Never'
    disp.RescaleTransferFunctionToDataRange(False, True)
    disp.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    for k, v in (('Title', title), ('ComponentTitle', ''), ('TitleFontSize', 16), ('LabelFontSize', 13),
                 ('AutomaticLabelFormat', 0), ('LabelFormat', fmt), ('AddRangeLabels', 0),
                 ('ScalarBarLength', 0.45), ('ScalarBarThickness', 14),
                 ('WindowLocation', 'Lower Right Corner')):
        try:
            setattr(bar, k, v)
        except (AttributeError, ValueError):
            pass   # property names differ slightly between ParaView versions

def pressure(src):
    # Cp = p / (0.5 U_inf^2): the solver's p is kinematic (p/rho) with p = 0 at the outlet
    if not SPEED:
        return src, lambda d: color(d, 'p', 'Kinematic pressure p (m2/s2)', ('Cool to Warm',))
    c = Calculator(Input=src)
    c.ResultArrayName = 'Cp'
    c.Function = 'p/%r' % (0.5 * SPEED ** 2)
    # strongest suction is full blue, the stagnation point full red
    return c, lambda d: color(d, 'Cp', 'Pressure coefficient Cp', ('Cool to Warm',))

def velocity(disp):
    color(disp, 'U', 'Velocity magnitude (m/s)', ('Turbo', 'Jet'), 'Magnitude', '%-#.1f')

def grey(disp):
    # solid colour without ColorBy(disp, None), which raises "invalid association string" on empty data
    disp.ColorArrayName = ['POINTS', '']
    disp.DiffuseColor = [0.75, 0.75, 0.78]
    disp.AmbientColor = [0.75, 0.75, 0.78]

def frame():
    # the box the camera frames: the part and its near wake, not the whole (much larger) domain
    ctr, L, ax, sgn = CFG['ctr'], max(CFG['ext']), CFG['axis'], CFG['sign']
    lo, hi = [], []
    for k in range(3):
        r = 0.75 * CFG['ext'][k] + 0.15 * L
        a, b = ctr[k] - r, ctr[k] + r
        if k == ax:
            a, b = (a, ctr[k] + 2 * L) if sgn > 0 else (ctr[k] - 2 * L, b)
        lo.append(a)
        hi.append(b)
    return lo, hi

def camera(direction, up):
    # placed by hand; render first, since paraview.simple resets the camera on a view's first Render()
    Render(view)
    lo, hi = frame()
    focus = [(lo[k] + hi[k]) / 2 for k in range(3)]
    radius = 0.5 * sum((hi[k] - lo[k]) ** 2 for k in range(3)) ** 0.5
    n = sum(d * d for d in direction) ** 0.5
    dist = radius / 0.2588 * 0.8   # sin(15 deg): ParaView's default 30 deg view angle; slightly tight
    view.CameraFocalPoint = focus
    view.CameraPosition = [focus[k] + dist * direction[k] / n for k in range(3)]
    view.CameraViewUp = up

def fine_slice(src, normal_axis, px=4000):
    # a plane through the part centre, sampled on a regular grid (px points across the framed region,
    # plus a margin, within the domain) instead of cut through the mesh: every point is interpolated
    # inside its cell, so coarse cells and refinement-level steps don't show as facets
    lo, hi = frame()
    side = max(hi[k] - lo[k] for k in range(3) if k != normal_axis)
    origin = list(CFG['ctr'])
    sym = CFG.get('sym')
    if sym and sym[0] == normal_axis and abs(origin[normal_axis] - sym[1]) < 1e-4 * max(CFG['ext']):
        origin[normal_axis] += 1e-3 * max(CFG['ext'])   # just off the symmetry plane, where the halves meet
    bounds, dims = [], []
    for k in range(3):
        if k == normal_axis:
            bounds += [origin[k]] * 2
            dims.append(1)
            continue
        m = 0.25 * (hi[k] - lo[k])
        a, b = max(lo[k] - m, CFG['lo'][k]), min(hi[k] + m, CFG['hi'][k])
        bounds += [a, b]
        dims.append(max(2, int(px * (b - a) / side)))
    # cut the plane first and sample that: probing a few thousand polygons, not the whole volume mesh
    s = Slice(Input=src)
    s.SliceType = 'Plane'
    try:
        s.Triangulatetheslice = 0   # one polygon per cut cell: no triangle fans to interpolate over
    except AttributeError:
        pass
    s.SliceType.Origin = origin
    s.SliceType.Normal = unit(normal_axis)
    # the cut points are stored in single precision, ~1e-10 m off the plane, and ResampleToImage then finds no
    # polygon at any sample (only a plane lying on mesh faces is exact): snap them onto the plane
    snap = Calculator(Input=s)
    snap.CoordinateResults = 1
    snap.Function = '+'.join(repr(origin[k]) + '*' + h if k == normal_axis else 'coords' + 'XYZ'[k] + '*' + h
                             for k, h in enumerate(('iHat', 'jHat', 'kHat')))
    r = ResampleToImage(Input=snap)
    r.UseInputBounds = 0
    r.SamplingBounds = bounds
    r.SamplingDimensions = dims
    t = Threshold(Input=r)   # drop the samples inside the part (outside the mesh)
    t.Scalars = ['POINTS', 'vtkValidPointMask']
    try:
        t.LowerThreshold, t.UpperThreshold = 0.5, 1.5
    except AttributeError:
        t.ThresholdRange = [0.5, 1.5]   # ParaView < 5.10
    return t

body_p, color_body_p = pressure(body)

if V == 'overview':
    Show(internal, view).SetRepresentationType('Outline')
    color_body_p(Show(body_p, view))
    view.ResetCamera()
else:
    ax, sgn, lift, pitch = CFG['axis'], CFG['sign'], CFG['lift_axis'], CFG['pitch_axis']
    ctr, ext = CFG['ctr'], CFG['ext']
    L = max(ext)
    # iso camera: from the side, a bit above, a bit upstream
    iso = [unit(pitch)[k] + 0.6 * unit(lift)[k] - 0.8 * sgn * unit(ax)[k] for k in range(3)]
    if V in ('uslice', 'pslice'):
        src, color_p = pressure(internal) if V == 'pslice' else (internal, None)
        s = fine_slice(src, pitch)   # plane contains the flow and lift directions
        if V == 'uslice':
            velocity(Show(s, view))
        else:
            color_p(Show(s, view))
        grey(Show(body, view))
        camera(unit(pitch), unit(lift))
    elif V == 'streamlines':
        try:
            st = StreamTracer(Input=internal, SeedType='Point Cloud')
        except Exception:
            st = StreamTracer(Input=internal, SeedType='Point Source')
        st.Vectors = ['POINTS', 'U']
        lo, hi = CFG['lo'], CFG['hi']
        size = [hi[k] - lo[k] for k in range(3)]
        seed = list(ctr)
        seed[ax] -= sgn * (0.5 * ext[ax] + 0.3 * L)   # just upstream of the part...
        # ...but inside the domain (a short upstream margin would put every seed outside the mesh)
        seed = [min(max(seed[k], lo[k] + 0.02 * size[k]), hi[k] - 0.02 * size[k]) for k in range(3)]
        st.SeedType.Center = seed
        st.SeedType.Radius = min(0.6 * max(ext[k] for k in range(3) if k != ax),
                                 0.45 * min(size[k] for k in range(3) if k != ax))
        st.SeedType.NumberOfPoints = 150
        st.MaximumStreamlineLength = size[ax]
        st.UpdatePipeline(scene.AnimationTime)
        lines = st
        traced = st.GetDataInformation().GetNumberOfPoints() > 0
        if traced and CFG.get('near'):
            # keep only the pathlines that come within CFG['near'] (m) of the part's surface. Plain VTK on
            # the fetched lines, not a ProgrammableFilter: in this builtin session that leaks numpy's
            # sum/min/max into the script's globals
            from vtkmodules.all import (vtkSTLReader, vtkImplicitPolyDataDistance, vtkIdList,
                                        vtkExtractCells, vtkGeometryFilter)
            stl = vtkSTLReader()
            stl.SetFileName('constant/geometry/part.stl')
            stl.Update()
            dist = vtkImplicitPolyDataDistance()
            dist.SetInput(stl.GetOutput())
            data = servermanager.Fetch(st)
            # a pathline is traced as two cells (forward and backward from its seed): keep both if either
            # comes near, so whole lines are shown
            seeds = data.GetCellData().GetArray('SeedIds')
            seed = (lambda c: seeds.GetValue(c)) if seeds else (lambda c: c)
            def gap(pt):
                # a half model's part.stl may be only the meshed half: measure the mirror side by its reflection
                d = abs(dist.EvaluateFunction(pt))
                if CFG.get('sym'):
                    q = list(pt)
                    q[CFG['sym'][0]] = 2 * CFG['sym'][1] - q[CFG['sym'][0]]
                    d = min(d, abs(dist.EvaluateFunction(q)))
                return d
            near = set()
            for c in range(data.GetNumberOfCells()):
                ids = data.GetCell(c).GetPointIds()
                if any(gap(data.GetPoint(ids.GetId(i))) <= CFG['near'] for i in range(ids.GetNumberOfIds())):
                    near.add(seed(c))
            keep = vtkIdList()
            for c in range(data.GetNumberOfCells()):
                if seed(c) in near:
                    keep.InsertNextId(c)
            ex = vtkExtractCells()
            ex.SetInputData(data)
            ex.SetCellList(keep)
            geo = vtkGeometryFilter()
            geo.SetInputConnection(ex.GetOutputPort())
            geo.Update()
            lines = TrivialProducer()
            lines.GetClientSideObject().SetOutput(geo.GetOutput())
            lines.UpdatePipeline()
        if lines.GetDataInformation().GetNumberOfPoints() == 0:
            # nothing traced (no flow field at this time, or no seed in the fluid), or nothing near the
            # part: show why, don't try to colour an empty result ("could not determine array range")
            if traced:
                print('PIFOAM: no pathlines pass within %g mm of the part - increase the distance.' % (CFG['near'] * 1000))
            else:
                print('PIFOAM: no pathlines could be traced - solve the case first, and give the domain room upstream.')
            Show(internal, view).SetRepresentationType('Outline')
            grey(Show(body, view))
        else:
            out = lines
            try:
                out = Tube(Input=lines)
                out.Radius = 0.004 * L
            except Exception:
                out = lines
            velocity(Show(out, view))
            grey(Show(body, view))
        camera(iso, unit(lift))
    elif V == 'bodyp':
        color_body_p(Show(body_p, view))
        camera(iso, unit(lift))
    elif V == 'wake':
        src, color_p = pressure(internal)
        calc = Calculator(Input=src)
        calc.ResultArrayName = 'Umag'
        calc.Function = 'mag(U)'
        c = Contour(Input=calc)
        c.ContourBy = ['POINTS', 'Umag']
        c.Isosurfaces = [0.5 * CFG['speed']]   # half free-stream speed: the wake / stagnation bubble
        cd = Show(c, view)
        color_p(cd)
        cd.Opacity = 0.6
        grey(Show(body, view))
        camera(iso, unit(lift))

Render()
"""


def paraview_script(case, view, near_mm=None):
    """pifoam_open.py for one case and preset: CFG header + PARAVIEW_SCRIPT.
    Geometry comes from compute_geometry() on the case's own (metre-scaled)
    part.stl and saved params; without those the script shows the overview.
    `near_mm` limits the streamlines view to pathlines passing that close to the part."""
    cfg = {"view": view}
    g = case_geometry(case)
    if g:   # without it (incomplete saved params) the script falls back to the overview
        cfg.update({k: g[k] for k in ("axis", "sign", "lift_axis", "pitch_axis", "ctr", "ext", "lo", "hi")},
                   speed=float(read_gui_json(case)["params"].get("speed", 1)))
        if g["sym"]:   # a half model: the script reflects the result, so frame the part and domain both sides
            k, c = g["sym"]
            mn, mx = min(g["mn"][k], 2 * c - g["mx"][k]), max(g["mx"][k], 2 * c - g["mn"][k])
            cfg["sym"] = g["sym"]
            cfg["ctr"][k], cfg["ext"][k] = (mn + mx) / 2, mx - mn
            cfg["lo"][k], cfg["hi"][k] = min(g["lo"][k], 2 * c - g["hi"][k]), max(g["hi"][k], 2 * c - g["lo"][k])
        if near_mm:
            cfg["near"] = near_mm / 1000
    return "CFG = " + json.dumps(cfg) + "\n" + PARAVIEW_SCRIPT


def case_geometry(case):
    """compute_geometry() for a written case, from its own (metre-scaled) part.stl and saved params;
    None when either is missing or the saved params are incomplete."""
    params = case_params(read_gui_json(case))
    stl = os.path.join(case, "constant/geometry/part.stl")
    if not params or not os.path.isfile(stl):
        return None
    try:
        return compute_geometry(read_stl(open(stl, "rb").read()), dict(params, scale="1", max_cells="1e18"))
    except (KeyError, ValueError):
        return None


def snappy_dict(level, s):
    """Render snappyHexMeshDict from the persisted per-case snappy settings
    (gui.json "snappy"), so setup and mesh auto-retry produce identical dicts
    apart from the refinement level."""
    loc = s["loc"]
    layers = int(s.get("layers", 0))
    expansion, min_thick = float(s.get("expansion", 1.2)), float(s.get("min_thick", 0.1))
    first = s.get("first_layer")   # absolute first-layer height from the y+ estimate (m), else relative sizing
    if first:
        thickness = (f"relativeSizes false; expansionRatio {expansion}; firstLayerThickness {first:g}; "
                     f"minThickness {min_thick * first:g};")
    else:
        thickness = (f"relativeSizes true; expansionRatio {expansion}; "
                     f"finalLayerThickness {float(s.get('final_layer', 0.3))}; minThickness {min_thick};")
    wake_level = int(s.get("wake_level", 0))
    wake_geom = wake_region = ""
    if wake_level > 0 and s.get("wake_min"):
        mn, mx = s["wake_min"], s["wake_max"]
        wake_geom = f"wake {{ type box; min ({mn[0]} {mn[1]} {mn[2]}); max ({mx[0]} {mx[1]} {mx[2]}); }}"
        wake_region = f"wake {{ mode inside; level {wake_level}; }}"
    return SNAPPY.substitute(
        level=level, lx=loc[0], ly=loc[1], lz=loc[2],
        addlayers="true" if layers > 0 else "false",
        layers=f"body {{ nSurfaceLayers {layers}; }}" if layers > 0 else "",
        thickness=thickness, wake_geom=wake_geom, wake_region=wake_region,
        maxcells=int(s.get("max_cells", 5000000)), nbetween=int(s.get("n_between", 3)),
        resolve=float(s.get("resolve_angle", 30)))


def field(cls, name, dims, internal, bf):
    out = FOAMFILE % (cls, name) + f"dimensions {dims};\ninternalField {internal};\nboundaryField\n{{\n"
    for patch, body in bf.items():
        out += f"    {patch} {{ {body} }}\n"
    return out + "}\n"


# ---------------------------------------------------------------- case setup
def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name or "")


MAX_FOLDER_DEPTH = 8   # folders inside folders, at most this deep


def case_rel(name):
    """A case's (or folder's) identifier relative to $FOAM_RUN: 'case' at the top level or 'folder/sub/case'
    inside folders, each part sanitized - the one way a name from the browser becomes a path."""
    parts = [sanitize_name(s.strip()) for s in str(name or "").strip().strip("/").split("/")]
    if not 1 <= len(parts) <= MAX_FOLDER_DEPTH + 1 or not all(parts) or any(set(s) == {"_"} for s in parts):
        raise ValueError(f"'{name}' is not a valid name (use 'case' or 'folder/case', folders at most "
                         f"{MAX_FOLDER_DEPTH} deep).")
    return "/".join(parts)


def folder_rel(name):
    """A folder's identifier: '' for the top level of $FOAM_RUN, else 'folder' or 'folder/sub'."""
    name = str(name or "").strip().strip("/")
    return case_rel(name) if name else ""


def rel_join(folder, leaf):
    return f"{folder}/{leaf}" if folder else leaf


def read_gui_json(case):
    try:
        return json.load(open(os.path.join(case, "gui.json")))
    except OSError:
        return {}


# form fields saved to an existing case as they're edited (no re-setup): only the inputs of the derived
# flight/stability figures, which never touch the case files. Everything else is a draft until re-created.
SAVED_PARAMS = ("mass", "mass_unit", "target_sm", "pv_near_mm")


def params_mass_kg(p):
    """The aircraft mass in kg from saved form params: 'mass' in 'mass_unit' (g | kg), or a legacy 'mass_kg'."""
    try:
        if "aircraft" in disabled_features(p):
            return 0.0
        if "mass" in p:
            return float(p.get("mass") or 0) * (0.001 if p.get("mass_unit") == "g" else 1)
        return float(p.get("mass_kg") or 0)
    except ValueError:
        return 0.0


def disabled_features(p):
    """The feature groups a case's params switch off (its 'features' param: space-separated FEATURE_GROUPS keys)."""
    off = str(p.get("features") or "").split()
    bad = [k for k in off if k not in FEATURE_GROUPS]
    if bad:
        raise ValueError(f"Unknown feature '{bad[0]}'. Features are {', '.join(FEATURE_GROUPS)}.")
    return off


def apply_features(p):
    """p with every disabled feature group's params reset to neutral, so a switched-off feature is really off."""
    out = dict(p)
    for k in disabled_features(p):
        out.update(FEATURE_GROUPS[k][1])
    return out


def case_params(info):
    """A case's saved form params (gui.json 'params') as they were used: switched-off features neutral. The
    saved params keep those features' values (the form shows them again); anything computing from them reads here."""
    params = info.get("params") if isinstance(info.get("params"), dict) else {}
    try:
        return apply_features(params)
    except ValueError:   # a hand-edited, unknown feature key: read it as saved rather than fail
        return dict(params)


def save_case_params(name, values, keys=SAVED_PARAMS):
    """Merge `keys` (default SAVED_PARAMS) from `values` into the case's saved form params (gui.json); False if
    the case has none yet. Sample lines are saved this way when they're resampled on an existing result."""
    path = os.path.join(case_path(name), "gui.json")
    info = read_gui_json(case_path(name))
    if not isinstance(info.get("params"), dict):
        return False
    info["params"].update({k: str(values[k]) for k in keys if k in values})
    json.dump(info, open(path, "w"))
    return True


def compute_geometry(raw_verts, p):
    """All of the pure geometry/mesh-sizing math that setup_case() needs, kept
    free of any file I/O so it can also back the no-side-effect /api/preview
    endpoint (shown live as soon as an STL is loaded, before 'Create case')."""
    scale = float(p["scale"])
    preset = MESH_PRESETS.get(p.get("mesh_preset"))
    legacy = LEGACY_PRESETS.get(p.get("mesh_preset", "custom"))
    direction = p["dir"]                       # e.g. "+y"
    sign = 1 if direction[0] == "+" else -1
    axis = "xyz".index(direction[1])

    # up (lift) axis: as the STL was exported; auto = z, or y when the flow runs along z
    up = p.get("up_axis") or "auto"
    if up not in ("auto", "x", "y", "z"):
        raise ValueError("Up axis must be auto, x, y or z.")
    lift_axis = (1 if axis == 2 else 2) if up == "auto" else "xyz".index(up)
    if lift_axis == axis:
        raise ValueError(f"The flow can't run along the up axis ({up.upper()}). Pick a horizontal flow direction.")
    pitch_axis = [i for i in range(3) if i not in (axis, lift_axis)][0]
    ground = p.get("ground") or "none"
    if ground not in GROUND_TYPES:
        raise ValueError(f"Ground must be one of {', '.join(GROUND_TYPES)}.")
    aoa, yaw = float(p.get("aoa") or 0), float(p.get("yaw") or 0)
    if abs(aoa) > MAX_ANGLE or abs(yaw) > MAX_ANGLE:
        raise ValueError(f"Angle of attack and yaw must be within ±{MAX_ANGLE}°.")
    if aoa and ground != "none":
        raise ValueError("An angle of attack tilts the flow into the ground. Use yaw, or turn the ground off.")
    half = p.get("half_model") or "none"
    if half not in HALF_MODELS:
        raise ValueError(f"Half model must be one of {', '.join(HALF_MODELS)}.")
    if half != "none" and yaw:
        raise ValueError("Yaw blows across the symmetry plane of a half model. Use the full model for yaw.")

    verts = [(x * scale, y * scale, z * scale) for x, y, z in raw_verts]
    mn, mx = bbox(verts)
    ext = [mx[i] - mn[i] for i in range(3)]
    L = max(ext)
    # half model: a symmetry plane across the span (pitch) axis at `root`, which becomes the domain's face there.
    # centre / custom cut the whole part and mesh the + side (the part sticks out through the plane, which
    # snappy trims); min / max are an already-half part, the plane just inside its root
    cut_min = half in ("centre", "custom", "min")   # the plane is the domain's min face on the span axis
    root = None
    if half in ("centre", "custom"):
        root = (mn[pitch_axis] + mx[pitch_axis]) / 2 if half == "centre" else opt_float(p, "half_mm")
        if root is None:
            raise ValueError("Give the cut plane's position in mm.")
        if half == "custom":
            root /= 1000
        if not mn[pitch_axis] < root < mx[pitch_axis]:
            raise ValueError(f"The cut plane must cross the part: between {mn[pitch_axis] * 1000:.4g} and "
                             f"{mx[pitch_axis] * 1000:.4g} mm on {'XYZ'[pitch_axis]}.")
    elif half != "none":
        root = (mn[pitch_axis] + HALF_INSET * ext[pitch_axis] if half == "min"
                else mx[pitch_axis] - HALF_INSET * ext[pitch_axis])
    domain_mode = p.get("domain_mode") or "relative"
    if domain_mode == "absolute":
        # the domain box given directly in metres (after the STL unit scaling); the ground, if any, is its
        # min face on the lift axis wherever that is
        try:
            lo = [float(p[f"dom_{c}0"]) for c in "xyz"]
            hi = [float(p[f"dom_{c}1"]) for c in "xyz"]
        except (KeyError, ValueError):
            raise ValueError("Give all six domain coordinates (X/Y/Z min and max, in metres).")
        if half in ("centre", "custom"):
            lo[pitch_axis] = root   # the cut plane, wherever the box's span-axis min was given
        for i, c in enumerate("XYZ"):
            # a floor (wheels) or a half model's symmetry plane (wing root) may cut into the part
            cut_lo = (i == lift_axis and ground != "none") or (i == pitch_axis and cut_min)
            cut_hi = i == pitch_axis and half == "max"
            if not (lo[i] < mn[i] or (cut_lo and lo[i] < mx[i])) or not (hi[i] > mx[i] or (cut_hi and hi[i] > mn[i])):
                raise ValueError(f"The domain must enclose the part: {c} needs min < {mn[i]:.4g} m and "
                                 f"max > {mx[i]:.4g} m.")
        if half != "none":
            root = lo[pitch_axis] if cut_min else hi[pitch_axis]
    elif domain_mode == "relative":
        side, up, down = (float(p["pad_side"]) * L, float(p["pad_up"]) * L, float(p["pad_down"]) * L)
        lo, hi = [], []
        for i in range(3):
            if i == axis:
                lo.append(mn[i] - (up if sign > 0 else down))
                hi.append(mx[i] + (down if sign > 0 else up))
            elif i == lift_axis and ground != "none":   # wind-tunnel floor at the part's lowest point (- clearance)
                lo.append(mn[i] - float(p.get("ground_gap_mm") or 0) / 1000)
                hi.append(mx[i] + side)
            elif i == pitch_axis and half != "none":   # symmetry plane at the root, free stream past the tip
                lo.append(root if cut_min else mn[i] - side)
                hi.append(mx[i] + side if cut_min else root)
            else:
                lo.append(mn[i] - side)
                hi.append(mx[i] + side)
    else:
        raise ValueError("Domain size must be 'relative' or 'absolute'.")
    if lo[lift_axis] >= mx[lift_axis]:
        raise ValueError("The ground would be above the part. Use a smaller ground clearance.")
    size = [hi[i] - lo[i] for i in range(3)]
    # background cell size h and surface level. Presets and "surface" (Custom) take a surface cell size in mm
    # and derive the level as the fewest halvings of h that reach it, so the surface cells are never coarser
    # than asked for; a preset sizes h from a background cell budget over this domain
    def surface_level(h, s):
        level = max(0, math.ceil(math.log2(h / s) - 1e-9))
        if level > MAX_LEVEL:
            raise ValueError(f"Reaching {s * 1000:g} mm from {h * 1000:.3g} mm cells needs refinement level {level}, "
                             f"more than {MAX_LEVEL}. Use a smaller background cell size.")
        return level
    vol = size[0] * size[1] * size[2]
    if preset:
        s = preset[1] / 1000
        h = max(s, (vol / preset[2]) ** (1 / 3))
        level = surface_level(h, s)
    elif legacy:
        h, level = L / legacy[0], legacy[1]
    elif p.get("mesh_preset") == "surface":
        try:
            h, s = float(p.get("cell_size_mm") or 0) / 1000, float(p.get("surface_cell_mm") or 0) / 1000
        except ValueError:
            h = s = 0
        if h <= 0 or s <= 0:
            raise ValueError("Background and surface cell sizes must be positive numbers of millimetres.")
        level = surface_level(h, s)
    elif p.get("mesh_preset") == "count":   # a cell-count sweep: level from 'level', else the surface size
        target = float(p.get("target_cells") or 0)
        if target < 1:
            raise ValueError("Target background cells must be a positive number.")
        h = (vol / target) ** (1 / 3)
        level = int(p["level"]) if p.get("level") else surface_level(h, float(p.get("surface_cell_mm") or 0) / 1000 or h / 8)
    elif "cell_size_mm" in p:
        mm = float(p["cell_size_mm"] or 0)
        if mm <= 0:
            raise ValueError("Background cell size must be a positive number of millimetres.")
        h, level = mm / 1000, int(p["level"])
    elif "cells_per_part" in p:   # legacy saved params: cells across the part
        h, level = L / float(p["cells_per_part"]), int(p["level"])
    else:                         # legacy saved params: raw background cell count
        h, level = (size[0] * size[1] * size[2] / float(p["cells"])) ** (1 / 3), int(p["level"])
    n = [max(4, round(size[i] / h)) for i in range(3)]
    max_cells = int(float(p.get("max_cells") or 5000000))
    if n[0] * n[1] * n[2] > max_cells:   # e.g. a mm size typed for a metre-scale part: refuse before blockMesh
        raise ValueError(f"The background mesh would have {n[0] * n[1] * n[2]:,} cells, more than Max cells "
                         f"({max_cells:,}). Use a larger cell size or fewer cells.")

    face = {0: ("(0 4 7 3)", "(1 2 6 5)"), 1: ("(0 1 5 4)", "(3 7 6 2)"), 2: ("(0 3 2 1)", "(4 5 6 7)")}
    mnface, mxface = face[axis][0], face[axis][1]
    inlet, outlet = (mnface, mxface) if sign > 0 else (mxface, mnface)
    ground_face = face[lift_axis][0] if ground != "none" else None
    sym_face = None if half == "none" else face[pitch_axis][0 if cut_min else 1]
    sides = " ".join(f for i in range(3) if i != axis for f in face[i] if f not in (ground_face, sym_face))

    # point in the air, upstream of the part
    loc = [(lo[i] + hi[i]) / 2 + 0.0123 * h for i in range(3)]
    upstream_gap = (mn[axis] - lo[axis]) if sign > 0 else (hi[axis] - mx[axis])
    loc[axis] = (lo[axis] + 0.5 * upstream_gap + 0.0123 * h) if sign > 0 else (hi[axis] - 0.5 * upstream_gap + 0.0123 * h)

    # flow direction tilted from the dir axis: angle of attack toward the lift axis, yaw toward the
    # pitch axis; lift/pitch directions orthogonalised against it (zero angles: the plain axes)
    a, b = math.radians(aoa), math.radians(yaw)
    e = lambda i: [1.0 if k == i else 0.0 for k in range(3)]
    flow = [math.cos(a) * math.cos(b) * sign * e(axis)[k] + math.sin(a) * e(lift_axis)[k]
            + math.cos(a) * math.sin(b) * e(pitch_axis)[k] for k in range(3)]

    def ortho(v, *basis):
        for u in basis:
            dot = sum(v[k] * u[k] for k in range(3))
            v = [v[k] - dot * u[k] for k in range(3)]
        n_ = math.sqrt(sum(c * c for c in v))
        return [c / n_ for c in v]

    lift_dir = ortho(e(lift_axis), flow)
    # pitch axis = lift x flow: a positive Cm turns the nose (upstream end) toward lift - nose-up, as for aircraft
    pitch_dir = [lift_dir[(k + 1) % 3] * flow[(k + 2) % 3] - lift_dir[(k + 2) % 3] * flow[(k + 1) % 3]
                 for k in range(3)]
    # reference values: auto = frontal bbox area and length along the flow (a half model counts both halves:
    # the meshed half and its reflection), custom = wing area, mean aerodynamic chord and the CG as moment
    # reference point
    span = (ext[pitch_axis] if half == "none" else 2 * ext[pitch_axis] if half in ("min", "max")
            else 2 * (mx[pitch_axis] - root))
    ctr = [(mn[i] + mx[i]) / 2 for i in range(3)]
    if (p.get("ref_mode") or "auto") == "custom":
        try:
            aref, lref = float(p["aref"]), float(p["lref"])
        except (KeyError, ValueError):
            raise ValueError("Give the reference area and length (wing area in m², mean chord in m).")
        if aref <= 0 or lref <= 0:
            raise ValueError("Reference area and length must be positive.")
        cofr = []
        for i, c in enumerate("xyz"):
            v = str(p.get(f"cofr_{c}") or "").strip()
            cofr.append(float(v) if v else ctr[i])   # blank: the part's centre on that axis
    else:
        aref, lref = ext[lift_axis] * span, ext[axis]   # lift and span are the two axes across the flow
        cofr = list(ctr)
    if root is not None:
        cofr[pitch_axis] = root   # on the symmetry plane, so the mirrored half's moments add up exactly

    # wake refinement box: from the part's downstream face, wake_len*L further
    # downstream, part's cross-section +/-25%, clipped to the domain
    wake_len = float(p.get("wake_len", 3)) * L
    wake_min, wake_max = [], []
    for i in range(3):
        if i == axis:
            a, b = (mx[i], mx[i] + wake_len) if sign > 0 else (mn[i] - wake_len, mn[i])
        else:
            a, b = mn[i] - 0.25 * ext[i], mx[i] + 0.25 * ext[i]
        wake_min.append(max(lo[i], a))
        wake_max.append(min(hi[i], b))

    nu = float(p.get("nu", 1.5e-5))
    speed = float(p.get("speed", 0))
    re = speed * lref / nu if nu > 0 else 0.0

    # y+ estimate (flat plate, turbulent: Cf = 0.026 Re^-1/7): first-layer height for the target y+, and
    # the layer count whose last layer grows to about half the surface cell size
    yplus = float(p.get("yplus") or 30)
    surface_cell = h / 2 ** level
    wake_level = int(float(p.get("wake_level") or 0))
    wake_cell = h / 2 ** wake_level if wake_level > 0 else None
    first_layer = layers_suggest = None
    if re > 0 and yplus > 0:
        u_tau = speed * math.sqrt(0.026 * re ** (-1 / 7) / 2)
        first_layer = yplus * nu / u_tau
        r, ratio = float(p.get("expansion") or 1.2), 0.5 * surface_cell / first_layer
        layers_suggest = 1 if ratio <= 1 or r <= 1 else min(30, math.ceil(1 + math.log(ratio) / math.log(r)))

    return {
        "verts": verts, "mn": mn, "mx": mx, "ext": ext, "axis": axis, "sign": sign,
        "lo": lo, "hi": hi, "size": size, "h": h, "n": n, "level": level,
        "inlet": inlet, "outlet": outlet, "sides": sides, "loc": loc,
        "lift_axis": lift_axis, "pitch_axis": pitch_axis,
        "aref": aref, "lref": lref, "ctr": ctr, "cofr": cofr, "span": span,
        "wake_min": wake_min, "wake_max": wake_max, "re": re,
        "flow": flow, "lift_dir": lift_dir, "pitch_dir": pitch_dir, "aoa": aoa, "yaw": yaw,
        "ground": ground, "ground_face": ground_face, "half": half, "sym_face": sym_face,
        "first_layer": first_layer, "surface_cell": surface_cell, "layers_suggest": layers_suggest,
        "wake_cell": wake_cell, "sym": None if root is None else [pitch_axis, root],
    }


def geometry_summary(g, p):
    """The compute_geometry() figures the UI shows, shared by /api/preview and /api/setup, plus the inflow
    they imply: dynamic pressure and the turbulence model's inlet values."""
    _, turb = inflow_turbulence(p)
    return {
        "background_cells": g["n"], "cell_size_m": g["h"], "level": g["level"],
        "domain_min": g["lo"], "domain_max": g["hi"],
        "Aref": g["aref"], "lRef": g["lref"], "Re": g["re"], "CofR": g["cofr"], "span": g["span"],
        "wake_min": g["wake_min"], "wake_max": g["wake_max"], "flow": g["flow"],
        "first_layer": g["first_layer"], "surface_cell": g["surface_cell"], "layers_suggest": g["layers_suggest"],
        "wake_cell": g["wake_cell"], "sym": g["sym"],
        "q":0.5 * float(p["rho"]) * float(p["speed"]) ** 2,
        "inlet_turb": {f: val for f, (_, val, _) in turb.items() if f != "nut"},
        "transient": transient_time(p, g) if is_transient(p) else None,
        "probes": parse_probes(p, g),
    }


def transient_time(p, g):
    """A transient case's time control (s), from the form and the geometry: end time (blank: 3 flow-throughs
    of the domain), write interval (end / write count), Courant limit and first time step (blank: half the
    Courant limit on a surface cell; the solver adjusts it from there)."""
    speed = float(p["speed"])
    if speed <= 0:
        raise ValueError("A transient run needs a positive flow speed.")
    flow_through = (g["hi"][g["axis"]] - g["lo"][g["axis"]]) / speed
    end = opt_float(p, "end_time_s") or 3 * flow_through
    count = opt_float(p, "write_count", 50)
    maxco = opt_float(p, "max_co", 1)
    dt = opt_float(p, "delta_t") or 0.5 * maxco * g["surface_cell"] / speed
    if end <= 0 or maxco <= 0 or dt <= 0 or count < 1:
        raise ValueError("End time, Courant number and time step must be positive, and at least one write.")
    write = end / int(count)
    return {"end": end, "write": write, "dt": min(dt, write), "maxco": maxco, "flow_through": flow_through,
            "steps": end / min(dt, write)}


def probe_groups(text, n, what):
    """'a b c; d e f' -> [(a, b, c), (d, e, f)] (commas count as spaces), each group exactly n numbers, mm -> m."""
    out = []
    for k, grp in enumerate(s for s in str(text or "").split(";") if s.strip()):
        try:
            nums = [float(c) / 1000 for c in grp.replace(",", " ").split()]
        except ValueError:
            nums = []
        if len(nums) != n:
            raise ValueError(f"{what} {k + 1} ('{grp.strip()}') needs {n} numbers in mm.")
        out.append(tuple(nums))
    return out


def parse_probes(p, g):
    """Point probes and sample lines from the form - mm in part coordinates (after the STL units) - in metres,
    each flagged whether it lies inside the domain: {points: [{name, at, inside}], lines: [{name, start, end,
    inside}], npts}. setup_case refuses probes outside; the preview just shows them."""
    lo, hi = g["lo"], g["hi"]
    inside = lambda c: all(lo[i] - 1e-9 <= c[i] <= hi[i] + 1e-9 for i in range(3))
    points = [{"name": f"P{k + 1}", "at": c, "inside": inside(c)}
              for k, c in enumerate(probe_groups(p.get("probe_points"), 3, "Probe point"))]
    lines = [{"name": f"line{k + 1}", "start": c[:3], "end": c[3:], "inside": inside(c[:3]) and inside(c[3:])}
             for k, c in enumerate(probe_groups(p.get("probe_lines"), 6, "Sample line"))]
    npts = int(opt_float(p, "probe_npts", 100))
    if not 2 <= npts <= 10000:
        raise ValueError("Points per sample line must be between 2 and 10000.")
    return {"points": points, "lines": lines, "npts": npts}


def probe_functions(probes):
    """controlDict function objects for parse_probes() output: 'probes' (U and p at the points, every time
    step: a time history) and 'lines' (U and p along each line, whenever fields are written)."""
    vec = lambda c: "(" + " ".join(f"{x:.10g}" for x in c) + ")"
    out = ""
    if probes["points"]:
        out += ("    probes\n    {\n        type            probes;\n        libs            (\"libsampling.so\");\n"
                "        writeControl    timeStep;\n        writeInterval   1;\n        fields          (U p);\n"
                "        probeLocations  (" + " ".join(vec(q["at"]) for q in probes["points"]) + ");\n    }\n")
    if probes["lines"]:
        out += lines_function(probes["lines"], probes["npts"])
    return out


def lines_function(lines, npts, write_control="writeTime"):
    vec = lambda c: "(" + " ".join(f"{x:.10g}" for x in c) + ")"
    sets = "".join(f"            {ln['name']} {{ type lineUniform; axis distance; start {vec(ln['start'])}; "
                   f"end {vec(ln['end'])}; nPoints {npts}; }}\n" for ln in lines)
    return ("    lines\n    {\n        type            sets;\n        libs            (\"libsampling.so\");\n"
            f"        writeControl    {write_control};\n        fields          (U p);\n"
            "        interpolationScheme cellPoint;\n        setFormat       raw;\n"
            "        sets\n        {\n" + sets + "        }\n    }\n")


def preview_geometry(p):
    """Non-destructive counterpart to setup_case(): same geometry/mesh-sizing
    math, no case folder touched - what the UI polls as soon as an STL loads
    and whenever a flow/mesh field changes."""
    if not STATE["verts"]:
        raise ValueError("Choose an STL file first.")
    p = apply_features(p)
    return geometry_summary(compute_geometry(STATE["verts"], p), p)


def tune_fvsolution(path, relax, targets, solve_fields=None):
    """Edit the tutorial's system/fvSolution in place: set entries of its flat
    'relaxationFactors' block and add 'residualControl' to its 'SIMPLE' block
    (neither block has nested braces in the motorBike tutorial). Empty dicts
    leave the tutorial's values untouched. solve_fields (the turbulence model's
    transported fields) replace nuTilda in the tutorial's "(U|nuTilda)" solver entry."""
    text = open(path).read()
    if solve_fields:
        text = text.replace('"(U|nuTilda)"', '"(' + "|".join(["U"] + list(solve_fields)) + ')"', 1)
    if relax:
        def relax_block(m):
            body = m.group(1)
            for k, v in relax.items():
                body, hits = re.subn(rf"(\n\s*{k}\s+)[^;]+;", lambda mm: f"{mm.group(1)}{v:g};", body)
                if not hits:
                    body += f"    {k} {v:g};\n"
            return "relaxationFactors\n{" + body + "}"
        text = re.sub(r"relaxationFactors\s*\{([^{}]*)\}", relax_block, text, count=1)
    if targets:
        rc = "    residualControl { " + " ".join(f"{k} {v:g};" for k, v in targets.items()) + " }\n"
        text = re.sub(r"(SIMPLE\s*\{[^{}]*)\}", lambda m: m.group(1) + rc + "}", text, count=1)
    open(path, "w").write(text)


def tune_fvschemes(path, fields, ddt=None):
    """Add a 'div(phi,<field>)' convection scheme for each transported turbulence field the tutorial's
    system/fvSchemes doesn't have yet (it covers nuTilda, k and omega; transition adds gammaInt, ReThetat).
    ddt: a time scheme (Euler | backward) replacing the tutorial's steadyState, for a transient run. The
    tutorial's 'bounded Gauss' convection schemes stay: the 'bounded' correction is harmless in time."""
    text = open(path).read()
    extra = "".join(f"\n    div(phi,{f}) bounded Gauss upwind;" for f in fields if f"div(phi,{f})" not in text)
    if extra:
        text = re.sub(r"(divSchemes\s*\{)", lambda m: m.group(1) + extra, text, count=1)
    if ddt:
        text, hits = re.subn(r"(ddtSchemes\s*\{\s*default\s+)\w+", lambda m: m.group(1) + ddt, text, count=1)
        if not hits:
            raise ValueError("Could not set the time scheme: the tutorial's fvSchemes has no ddtSchemes default.")
    open(path, "w").write(text)


def transient_fvsolution(path, fields, n_outer, n_corr):
    """Turn the tutorial's steady fvSolution into a PIMPLE one (as OpenFOAM 13's transient pitzDaily): the
    'SIMPLE' block becomes 'PIMPLE' with the corrector counts, the transported fields get '...Final' solver
    entries (the last outer corrector, solved tighter; pFinal is already there) and relaxation is dropped."""
    text = open(path).read()
    pimple = (f"PIMPLE\n{{\n    nOuterCorrectors {n_outer};\n    nCorrectors     {n_corr};\n"
              "    nNonOrthogonalCorrectors 0;\n}")
    text, hits = re.subn(r"SIMPLE\s*\{[^{}]*\}", lambda m: pimple, text, count=1)
    if not hits:
        raise ValueError("Could not set up PIMPLE: the tutorial's fvSolution has no SIMPLE block.")
    text = re.sub(r"relaxationFactors\s*\{[^{}]*\}\s*", "", text, count=1)
    final = ('    "(' + "|".join(fields) + ')Final"\n    {\n        solver          smoothSolver;\n'
             "        smoother        GaussSeidel;\n        tolerance       1e-08;\n        relTol          0;\n    }\n")
    text = re.sub(r"(solvers\s*\{)", lambda m: m.group(1) + "\n" + final, text, count=1)
    open(path, "w").write(text)


def turb_fields(model, nu, speed, ratio, intensity, length=None):
    """Turbulence fields of a RAS model: {name: (dimensions, free-stream value, wall condition)}. Inlet,
    outlet, sides, ground and symmetry conditions follow from these in setup_case, alike for every field.
    ratio: inlet turbulent/laminar viscosity ratio, intensity: inlet turbulence intensity (fraction),
    length: inlet turbulence length scale (m) - when given, omega follows from it instead of from ratio."""
    if model == "SpalartAllmaras":
        return {"nuTilda": ("[0 2 -1 0 0 0 0]", ratio * nu, "type fixedValue; value uniform 0;"),
                "nut": ("[0 2 -1 0 0 0 0]", 0, "type nutUSpaldingWallFunction; value uniform 0;")}
    k = 1.5 * (speed * intensity) ** 2
    omega = math.sqrt(k) / (0.09 ** 0.25 * length) if length else k / (nu * ratio)
    out = {"k": ("[0 2 -2 0 0 0 0]", k, "type kqRWallFunction; value uniform %g;" % k),
           "omega": ("[0 0 -1 0 0 0 0]", omega, "type omegaWallFunction; value uniform %g;" % omega),
           "nut": ("[0 2 -1 0 0 0 0]", 0, "type nutkWallFunction; value uniform 0;")}
    if model == "kOmegaSSTLM":
        # wall-resolved (y+ ~ 1, as in OpenFOAM's T3A tutorial): k is zero at the wall; transition onset
        # Re_theta,t from the free-stream intensity (Langtry-Menter correlation, Tu in %, kept >= 0.027 %)
        tu = max(intensity * 100, 0.027)
        re_t = 1173.51 - 589.428 * tu + 0.2196 / tu ** 2 if tu <= 1.3 else 331.5 * (tu - 0.5658) ** -0.671
        out["k"] = ("[0 2 -2 0 0 0 0]", k, "type fixedValue; value uniform 0;")
        out["gammaInt"] = ("[0 0 0 0 0 0 0]", 1, "type zeroGradient;")
        out["ReThetat"] = ("[0 0 0 0 0 0 0]", re_t, "type zeroGradient;")
    return out


def opt_float(p, key, default=None):
    """An optional numeric form field: blank or missing -> default."""
    v = p.get(key)
    return default if v is None or str(v).strip() == "" else float(v)


def inflow_turbulence(p):
    """(model, turb_fields) for the form params p - the one place setup_case and the preview read the
    turbulence model and its inflow settings from."""
    model = p.get("turb_model") or "SpalartAllmaras"
    if model not in TURB_MODELS:
        raise ValueError(f"Turbulence model must be one of {', '.join(TURB_MODELS)}.")
    ratio, intensity = opt_float(p, "nut_ratio", 3), opt_float(p, "turb_intensity", 0.1) / 100
    if ratio <= 0 or intensity <= 0:
        raise ValueError("Inlet viscosity ratio and turbulence intensity must be positive.")
    length = None   # k-omega models: omega from a turbulence length scale (mm) instead of the viscosity ratio
    if (p.get("omega_mode") or "ratio") == "length" and model != "SpalartAllmaras":
        length = opt_float(p, "turb_length_mm")
        if not length or length <= 0:
            raise ValueError("The inlet turbulence length scale must be a positive number of mm.")
        length /= 1000
    return model, turb_fields(model, float(p.get("nu", 1.5e-5)), float(p["speed"]), ratio, intensity, length)


def setup_case(p, verts=None):
    """verts: geometry captured when a queued sweep was submitted; defaults to
    the currently uploaded STL."""
    raw_verts = verts or STATE["verts"]
    if not raw_verts:
        raise ValueError("Choose an STL file first.")
    raw, p = p, apply_features(p)   # written with switched-off features neutral; their values are still saved
    if p.get("sim_type", "external") not in SIM_TYPES:
        raise ValueError(f"Unknown simulation type '{p.get('sim_type')}'.")
    name = case_rel(p.get("case") or "gui_case")
    case = os.path.join(RUN, name)
    if is_folder(case):   # re-creating a case replaces its directory: never a folder of cases
        raise ValueError(f"'{name}' is a folder, not a case. Pick another case name.")
    if "/" in name and not is_folder(os.path.dirname(case)):
        raise ValueError(f"Folder '{name.rpartition('/')[0]}' does not exist. Create it on the home screen first.")
    speed = float(p["speed"])
    rho = float(p["rho"])
    nu = float(p.get("nu", 1.5e-5))
    if nu <= 0 or rho <= 0:
        raise ValueError("Density and viscosity must be positive.")
    side_bc = p.get("side_bc") or "slip"
    if side_bc not in ("slip", "symmetry", "freestream"):
        raise ValueError("Side boundary must be 'slip', 'symmetry' or 'freestream'.")
    layer_mode = p.get("layer_mode") or "relative"
    if layer_mode not in ("relative", "yplus"):
        raise ValueError("Layer sizing must be 'relative' or 'yplus'.")
    opt = lambda key, default=None: opt_float(p, key, default)   # advanced fields: blank or missing -> default
    model, turb = inflow_turbulence(p)
    transported = [f for f in turb if f != "nut"]
    # 'Relaxation turbulence' (saved as relax_nuTilda) applies to the model's transported fields;
    # Spalart-Allmaras keeps the tutorial's nuTilda value when blank, the k-omega fields default to 0.5
    relax = {"U": opt("relax_U"), "p": opt("relax_p")}
    relax.update({f: opt("relax_nuTilda", None if model == "SpalartAllmaras" else 0.5) for f in transported})
    targets = {k: opt("res_" + k) for k in ("p", "U")}
    if any(v is not None and not 0 < v <= 1 for v in relax.values()):
        raise ValueError("Relaxation factors must be in (0, 1].")
    if any(v is not None and v <= 0 for v in targets.values()):
        raise ValueError("Residual targets must be positive.")
    transient = is_transient(p)
    scheme = p.get("time_scheme") or "Euler"
    n_outer, n_corr = int(opt("n_outer", 1)), int(opt("n_corr", 2))
    if transient and (scheme not in ("Euler", "backward") or not 1 <= n_outer <= 50 or not 1 <= n_corr <= 10):
        raise ValueError("Time scheme must be Euler or backward, outer correctors 1-50 and pressure correctors 1-10.")
    init = (p.get("init_case") or "").strip() if transient else ""
    if init:   # warm start: the source must have a solution to map from
        src = case_path(init)
        if case_rel(init) == name:
            raise ValueError("A case can't be initialised from itself.")
        if not any(_num(t) for t in os.listdir(src)):
            raise ValueError(f"Case '{init}' has no solution to start from yet.")

    g = compute_geometry(raw_verts, p)
    probes = parse_probes(p, g)
    outside = [q["name"] for q in probes["points"] + probes["lines"] if not q["inside"]]
    if outside:
        raise ValueError(f"Probe{'s' if len(outside) > 1 else ''} {', '.join(outside)} lie{'' if len(outside) > 1 else 's'} "
                         "outside the flow domain.")
    tt = transient_time(p, g) if transient else None
    level = g["level"]
    verts, axis, sign = g["verts"], g["axis"], g["sign"]
    lo, hi, n = g["lo"], g["hi"], g["n"]
    inlet, outlet, sides, loc = g["inlet"], g["outlet"], g["sides"], g["loc"]
    aref, lref, cofr = g["aref"], g["lref"], g["cofr"]
    if (g["aoa"] or g["yaw"]) and side_bc != "freestream":
        # a tilted inflow crosses the side faces; slip/symmetry walls would force it back along the axis
        raise ValueError("A non-zero angle of attack or yaw needs Freestream side walls.")
    if layer_mode == "yplus" and not g["first_layer"]:
        raise ValueError("The y+ layer sizing needs a positive flow speed and viscosity.")

    # build case from the motorBike tutorial (keeps working solver/turbulence settings)
    if os.path.exists(case):
        shutil.rmtree(case)
    shutil.copytree(TUTORIAL, case)
    for d in ("0", "constant/polyMesh", "constant/geometry", "constant/triSurface", "postProcessing"):
        shutil.rmtree(os.path.join(case, d), ignore_errors=True)
    for fpath in ("system/functions", "Allrun", "Allclean"):
        try:
            os.remove(os.path.join(case, fpath))
        except OSError:
            pass
    os.makedirs(os.path.join(case, "0"))
    os.makedirs(os.path.join(case, "constant/geometry"))
    # the whole part, also when halved: snappy keeps the half inside the domain, ParaView reflects the result
    write_ascii_stl(os.path.join(case, "constant/geometry/part.stl"), verts, "part")

    w = lambda rel, text: open(os.path.join(case, rel), "w").write(text)
    w("system/blockMeshDict", BLOCKMESH.substitute(
        x0=lo[0], y0=lo[1], z0=lo[2], x1=hi[0], y1=hi[1], z1=hi[2],
        nx=n[0], ny=n[1], nz=n[2], inlet=inlet, outlet=outlet, sides=sides,
        sidetype="symmetry" if side_bc == "symmetry" else "patch",   # 'slip' is a field BC, not a patch type
        extrapatches=(f"ground {{ type wall; faces ({g['ground_face']}); }}" if g["ground_face"] else "") +
        (f"\n    symmetry {{ type symmetryPlane; faces ({g['sym_face']}); }}" if g["sym_face"] else "")))
    w("system/surfaceFeaturesDict", FEATURES.substitute(angle=opt("feature_angle", 150)))
    snappy = {"loc": loc, "layers": int(p.get("layers", 0)), "expansion": float(p.get("expansion", 1.2)),
              "wake_level": int(p.get("wake_level", 0)),
              "wake_min": g["wake_min"], "wake_max": g["wake_max"],
              "n_between": int(opt("n_between", 3)), "resolve_angle": opt("resolve_angle", 30),
              "max_cells": int(opt("max_cells", 5000000)), "final_layer": opt("final_layer", 0.3),
              "min_thick": opt("min_thick", 0.1),
              "first_layer": g["first_layer"] if layer_mode == "yplus" else None}
    w("system/snappyHexMeshDict", snappy_dict(level, snappy))
    w("constant/physicalProperties", PHYSICAL % nu)
    fvsol = os.path.join(case, "system/fvSolution")
    if transient:   # PIMPLE time stepping: no relaxation, no residual stop (both are steady-state tools)
        tune_fvsolution(fvsol, {}, {}, None if model == "SpalartAllmaras" else transported)
        transient_fvsolution(fvsol, ["U"] + transported, n_outer, n_corr)
    else:
        tune_fvsolution(fvsol, {k: v for k, v in relax.items() if v is not None},
                        {k: v for k, v in targets.items() if v is not None},
                        None if model == "SpalartAllmaras" else transported)
    tune_fvschemes(os.path.join(case, "system/fvSchemes"), transported, scheme if transient else None)
    mt = os.path.join(case, "constant/momentumTransport")
    text = re.sub(r"(\bmodel\s+)\w+\s*;", lambda m: m.group(1) + model + ";", open(mt).read(), count=1)
    open(mt, "w").write(text)   # read before opening for write, which truncates

    vec = lambda c: "(" + " ".join(f"{x:.10g}" for x in c) + ")"
    v = vec([speed * c for c in g["flow"]])
    belt = vec([speed * sign if k == axis else 0.0 for k in range(3)])   # moving ground: along the dir axis
    # per field: class, dimensions, internal value and the patch conditions. sides: 'slip' (default), a
    # 'symmetry' patch every field must match, or far-field 'freestream' (in/outflow decided per face by the
    # flow direction - needed once the inflow is tilted). ground: a wall, fixed or moving with the free stream
    fields = {
        "U": ("volVectorField", "[0 1 -1 0 0 0 0]", f"uniform {v}", {
            "inlet": f"type fixedValue; value uniform {v};",
            "outlet": f"type inletOutlet; inletValue uniform (0 0 0); value uniform {v};",
            "body": "type noSlip;",
            "sides": {"slip": "type slip;", "symmetry": "type symmetry;",
                      "freestream": f"type freestreamVelocity; freestreamValue uniform {v};"}[side_bc],
            "ground": "type noSlip;" if g["ground"] == "fixed" else f"type fixedValue; value uniform {belt};"}),
        "p": ("volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0", {
            "inlet": "type zeroGradient;", "outlet": "type fixedValue; value uniform 0;",
            "body": "type zeroGradient;",
            "sides": {"slip": "type zeroGradient;", "symmetry": "type symmetry;",
                      "freestream": "type freestreamPressure; freestreamValue uniform 0;"}[side_bc],
            "ground": "type zeroGradient;"}),
    }
    for fname, (dims, val, wall) in turb.items():
        u, calc = f"uniform {val:g}", fname == "nut"   # nut is computed by the model, not transported
        fields[fname] = ("volScalarField", dims, u, {
            "inlet": "type calculated; value uniform 0;" if calc else f"type fixedValue; value {u};",
            "outlet": "type calculated; value uniform 0;" if calc else f"type inletOutlet; inletValue {u}; value {u};",
            "body": wall,
            "sides": {"slip": "type slip;", "symmetry": "type symmetry;",
                      "freestream": "type calculated; value uniform 0;" if calc
                      else f"type freestream; freestreamValue {u};"}[side_bc],
            "ground": wall})
    for fname, (cls, dims, internal, bf) in fields.items():
        if not g["ground_face"]:
            del bf["ground"]
        if g["sym_face"]:
            bf["symmetry"] = "type symmetryPlane;"
        w("0/" + fname, field(cls, fname, dims, internal, bf))

    zero = sorted(os.listdir(os.path.join(case, "0")))
    if transient:
        endtime, final = f"{tt['end']:.6g}", ""   # adjustableRunTime writes exactly at endTime anyway
        timecontrol = TRANSIENT_TIME.substitute(deltat=f"{tt['dt']:.6g}", writeinterval=f"{tt['write']:.6g}",
                                                maxco=f"{tt['maxco']:g}")
    else:
        endtime = int(p["endtime"])
        timecontrol = STEADY_TIME.substitute(writeinterval=max(1, int(opt("write_interval", 100))))
        final = FINAL_FIELDS.substitute(endtime=endtime, writefields=" ".join(zero))
    w("system/controlDict", CONTROL.substitute(
        endtime=endtime, timecontrol=timecontrol, finalfields=final, probes=probe_functions(probes),
        rho=rho, cx=cofr[0], cy=cofr[1], cz=cofr[2],
        lift=vec(g["lift_dir"])[1:-1], drag=vec(g["flow"])[1:-1], pitch=vec(g["pitch_dir"])[1:-1],
        # a half model's forces are those of one half: half the area gives full-aircraft coefficients
        speed=speed, lref=lref, aref=aref / 2 if g["sym_face"] else aref,
        resfields=" ".join(f for f in zero if f != "nut")))

    info = {"case": name, "speed": speed, "rho": rho, "nu": nu, "aref": aref, "lref": lref,
            "cofr": cofr, "span": g["span"], "turb_model": model,
            "half_axis": g["pitch_axis"] if g["sym_face"] else None,
            "level": level, "loc": loc, "snappy": snappy, "params": dict(raw, case=name),
            "end_time": tt["end"] if tt else int(p["endtime"])}   # iterations (steady) or seconds (transient)
    json.dump(info, open(os.path.join(case, "gui.json"), "w"))
    return {"case": name, "path": case, **geometry_summary(g, p)}


def case_path(name):
    path = os.path.join(RUN, case_rel(name))
    if not is_case(path):
        raise ValueError(f"Case '{name}' does not exist yet. Click 'Create case' first.")
    return path


def is_case(path):
    return os.path.isfile(os.path.join(path, "system", "controlDict"))


FOLDER_META = "project.json"   # marks a folder of cases (named from when folders were one-level "projects")
FOLDER_STL = "geometry.stl"    # a folder's shared part, as uploaded (its units in FOLDER_META 'scale')


def is_folder(path):
    """A folder of cases is a directory holding FOLDER_META, at the top of $FOAM_RUN or inside another folder;
    other directories are left alone."""
    return os.path.isfile(os.path.join(path, FOLDER_META))


def read_folder(path):
    try:
        return json.load(open(os.path.join(path, FOLDER_META)))
    except (OSError, ValueError):
        return {}


def case_entry(rel, path):
    """One case for the home screen: state, type and key figures."""
    has_results = os.path.isdir(os.path.join(path, "postProcessing/forceCoeffs"))
    coeffs = {}
    if has_results:   # the home screen's key figures: mean coefficients over the case's default window
        try:
            coeffs = {k: v["mean"] for k, v in case_averages(path, parse_coeffs_series(path)).items()
                      if k in ("Cd", "Cl")}
        except Exception:
            pass
    mtime = max([os.path.getmtime(path)] + [os.path.getmtime(os.path.join(path, f))
                                            for f in ("gui.json", "mesh_status.json", "postProcessing")
                                            if os.path.exists(os.path.join(path, f))])
    info = read_gui_json(path)
    return {
        "name": rel,
        "folder": rel.rpartition("/")[0],
        "sim_type": (info.get("params") or {}).get("sim_type", "external"),
        "turb_model": info.get("turb_model"),
        "mtime": mtime,
        "has_mesh": os.path.isfile(os.path.join(path, "constant/polyMesh/owner")),
        "has_results": has_results,
        "has_stl": os.path.isfile(os.path.join(path, "constant/geometry/part.stl")),
        "cells": (read_mesh_status(path).get("stats") or {}).get("cells"),
        **coeffs,
    }


def walk(folder=""):
    """('case' | 'folder', name, path) for everything under a folder ('' = all of $FOAM_RUN), depth first."""
    base = os.path.join(RUN, folder) if folder else RUN
    if not os.path.isdir(base):
        return
    for entry in sorted(os.listdir(base)):
        rel, path = rel_join(folder, entry), os.path.join(base, entry)
        if is_case(path):
            yield "case", rel, path
        elif is_folder(path):
            yield "folder", rel, path
            yield from walk(rel)


def case_dirs(folder=""):
    """(name, path) of every case under a folder ('' = all): 'case' at the top level, 'folder/sub/case' inside."""
    return ((rel, path) for kind, rel, path in walk(folder) if kind == "case")


def list_cases():
    return sorted((case_entry(rel, path) for rel, path in case_dirs()), key=lambda c: -c["mtime"])


def list_folders():
    """Every folder: its parent, direct cases and subfolders, all cases below it, and its shared part."""
    items = list(walk())
    out = []
    for kind, rel, path in items:
        if kind != "folder":
            continue
        meta, has_stl = read_folder(path), os.path.isfile(os.path.join(path, FOLDER_STL))
        inside = [(k, r) for k, r, _ in items if r.startswith(rel + "/")]
        out.append({"name": rel, "parent": rel.rpartition("/")[0], "mtime": os.path.getmtime(path),
                    "cases": sum(k == "case" and r.count("/") == rel.count("/") + 1 for k, r in inside),
                    "folders": sum(k == "folder" and r.count("/") == rel.count("/") + 1 for k, r in inside),
                    "all_cases": sum(k == "case" for k, _ in inside),
                    "has_stl": has_stl, "stl": meta.get("stl") if has_stl else None,
                    "scale": meta.get("scale") or "0.001"})
    return out


def retarget_refs(renamed):
    """After cases were moved or renamed ({old name: new name}), point other cases' references to them (a
    transient case's warm-start source, init_case) at the new names, so they keep working."""
    for _, path in case_dirs():
        info = read_gui_json(path)
        params = info.get("params") if isinstance(info.get("params"), dict) else {}
        if params.get("init_case") in renamed:
            params["init_case"] = renamed[params["init_case"]]
            json.dump(info, open(os.path.join(path, "gui.json"), "w"))


def folder_dir(name, parent_ok=False):
    """(name, path) of an existing folder; parent_ok also accepts '' (the top level of $FOAM_RUN)."""
    rel = folder_rel(name)
    if not rel and parent_ok:
        return "", RUN
    if not rel:
        raise ValueError("Give the folder a name.")
    path = os.path.join(RUN, rel)
    if not is_folder(path):
        raise ValueError(f"Folder '{rel}' does not exist.")
    return rel, path


def folder_leaf(name, what="folder"):
    """A folder's (or case's) own name, sanitised; refuses one that's empty or only underscores."""
    leaf = sanitize_name(str(name or "").strip())
    if not leaf or set(leaf) == {"_"}:
        raise ValueError(f"Give the {what} a name (letters, digits and underscores).")
    return leaf


def case_busy(rel):
    """Whether a running or queued job uses this case or (for a folder) any case below it. A sweep is queued
    under its base case but creates and runs '<base>_sweepN' variants next to it."""
    def uses(job_case, kind):
        if not job_case:
            return False
        return job_case == rel or job_case.startswith(rel + "/") or \
            (kind == "sweep" and re.fullmatch(re.escape(job_case) + r"_sweep\d+", rel) is not None)
    # JOB["case"] is the running sweep variant; the queue item that started it keeps the base name
    if JOB["running"] and (uses(JOB["case"], JOB["kind"]) or uses(JOB.get("base"), JOB["kind"])):
        return True
    return any(uses(item["case"], item["kind"]) for item in queue_snapshot())


def busy_error(rel):
    return f"'{rel}' is in use by a running or queued job. Stop it or remove it from the queue first."


def subtree_depth(rel, path):
    """How many folder levels a folder and everything below it spans (a folder holding only cases: 1)."""
    return 1 + max([r.count("/") - rel.count("/") for k, r, _ in walk(rel) if k == "folder"] or [0])


def create_folder(name, parent=""):
    parent, base = folder_dir(parent, parent_ok=True)
    rel = rel_join(parent, folder_leaf(name))
    if rel.count("/") + 1 > MAX_FOLDER_DEPTH:
        raise ValueError(f"Folders can be nested at most {MAX_FOLDER_DEPTH} deep.")
    path = os.path.join(RUN, rel)
    if os.path.exists(path):
        raise ValueError(f"'{rel}' already exists.")
    os.makedirs(path)
    json.dump({"created": time.time(), "stl": None, "scale": "0.001"}, open(os.path.join(path, FOLDER_META), "w"))
    return rel


def relocate(old, new):
    """After a folder moved from name `old` to `new`: every case below it records its new name, and other
    cases' references to them follow (retarget_refs)."""
    renamed = {}
    for rel, path in case_dirs(new):
        set_gui_case(path, rel)
        renamed[old + rel[len(new):]] = rel
    retarget_refs(renamed)
    THUMBS.clear()


def move_folder(name, dest="", leaf=None):
    """Move a folder into another folder ('' = the top level), optionally renaming it (leaf). Returns its new
    name. A folder can't go inside itself, and the result must stay within MAX_FOLDER_DEPTH."""
    rel, path = folder_dir(name)
    dest, _ = folder_dir(dest, parent_ok=True)
    if dest == rel or dest.startswith(rel + "/"):
        raise ValueError("A folder can't be moved into itself.")
    new = rel_join(dest, folder_leaf(leaf) if leaf is not None else rel.rpartition("/")[2])
    if new == rel:
        return rel
    if new.count("/") + subtree_depth(rel, path) > MAX_FOLDER_DEPTH:
        raise ValueError(f"That would nest folders more than {MAX_FOLDER_DEPTH} deep.")
    if os.path.exists(os.path.join(RUN, new)):
        raise ValueError(f"'{new}' already exists.")
    if case_busy(rel):
        raise ValueError(busy_error(rel))
    os.rename(path, os.path.join(RUN, new))
    relocate(rel, new)
    return new


def rename_folder(name, new):
    rel, _ = folder_dir(name)
    return move_folder(rel, rel.rpartition("/")[0], leaf=new)


def delete_folder(name, with_cases=False):
    rel, path = folder_dir(name)
    n = sum(1 for _ in case_dirs(rel))
    if n and not with_cases:
        raise ValueError(f"Folder '{rel}' still holds {n} case{'s' if n != 1 else ''} (counting subfolders).")
    if case_busy(rel):
        raise ValueError(busy_error(rel))
    shutil.rmtree(path)
    THUMBS.clear()


def set_gui_case(path, rel):
    info = read_gui_json(path)
    if info:
        info["case"] = rel
        if isinstance(info.get("params"), dict):
            info["params"]["case"] = rel
        json.dump(info, open(os.path.join(path, "gui.json"), "w"))


def move_case(name, folder, leaf=None):
    """Move a case into a folder ('' = the top level), optionally renaming it (leaf). Returns its new name."""
    src = case_path(name)
    rel = case_rel(name)
    new = rel_join(folder_dir(folder, parent_ok=True)[0],
                   folder_leaf(leaf, "case") if leaf is not None else rel.rpartition("/")[2])
    if new == rel:
        return rel
    dst = os.path.join(RUN, new)
    if os.path.exists(dst):
        raise ValueError(f"'{new}' already exists.")
    if case_busy(rel):
        raise ValueError(busy_error(rel))
    os.rename(src, dst)
    set_gui_case(dst, new)
    retarget_refs({rel: new})
    return new


def rename_case(name, new):
    """Rename a case in place (same folder): the directory, its gui.json and other cases' references to it."""
    rel = case_rel(name)
    return move_case(rel, rel.rpartition("/")[0], leaf=new)


def set_folder_stl(name, data, filename, scale):
    rel, path = folder_dir(name)
    if len(read_stl(data)) < 3:
        raise ValueError("Could not read triangles from that STL.")
    float(scale)   # validated: the units it was drawn in
    open(os.path.join(path, FOLDER_STL), "wb").write(data)
    meta = read_folder(path)
    meta.update(stl=os.path.basename(filename or FOLDER_STL), scale=str(scale))
    json.dump(meta, open(os.path.join(path, FOLDER_META), "w"))


def folder_stl(name):
    """The shared part that applies inside a folder: its own, else the nearest enclosing folder's -> (path,
    meta) or (None, None). New cases anywhere below a folder with a part start from it."""
    rel = folder_rel(name)
    while rel:
        path = os.path.join(RUN, rel)
        if os.path.isfile(os.path.join(path, FOLDER_STL)):
            return os.path.join(path, FOLDER_STL), read_folder(path)
        rel = rel.rpartition("/")[0]
    return None, None


def use_folder_stl(name):
    """Load the shared part that applies in a folder as the current geometry (like an upload), with its units."""
    stl, meta = folder_stl(folder_dir(name)[0])
    if not stl:
        raise ValueError(f"Folder '{folder_rel(name)}' has no shared part.")
    return {**load_stl(open(stl, "rb").read(), meta.get("stl") or FOLDER_STL, stl), "scale": meta.get("scale") or "0.001"}


def load_stl(data, name, stl=None):
    """Make an STL the current geometry (STATE) -> the upload response: size, extents and the preview.
    stl: the file the bytes came from, so its preview comes from the disk cache (stl_preview_cached)."""
    verts = read_stl(data)
    if len(verts) < 3:
        raise ValueError("Could not read triangles from that STL.")
    STATE["verts"], STATE["name"] = verts, name
    mn, mx = bbox(verts)
    return {"tris": len(verts) // 3, "min": mn, "max": mx, "ext": [mx[i] - mn[i] for i in range(3)],
            "preview": stl_preview_cached(stl, verts) if stl else stl_preview(verts), "name": name}


def stl_preview_cached(stl, verts=None):
    """stl_preview of an STL file, cached on disk beside it (<stl>.preview.json, valid while the file's size and
    mtime and PREVIEW_TRIS are unchanged): decimating a large part takes seconds, every time a case is opened.
    verts: the file's vertices if already read; otherwise it's read only when the cache is stale."""
    st = os.stat(stl)
    key, side = [st.st_size, st.st_mtime, PREVIEW_TRIS], stl + ".preview.json"
    try:
        with open(side) as f:
            cached = json.load(f)
        if cached.get("key") == key:
            return cached["preview"]
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    pv = stl_preview(verts if verts is not None else read_stl(open(stl, "rb").read()))
    try:
        with open(side, "w") as f:
            json.dump({"key": key, "preview": pv}, f)
    except OSError:
        pass   # a read-only case still opens, just without the cache
    return pv


THUMBS = {}   # STL path -> (mtime, decimated preview): home-screen thumbnails, built once per STL


def stl_thumb(stl, max_tris=5000):
    """A small decimated copy of an STL (a case's part.stl, a folder's shared part) for a home-screen tile
    (cached until the STL changes; OSError when there's none). Decimated from the cached preview."""
    mtime = os.path.getmtime(stl)
    if THUMBS.get(stl, (None,))[0] != mtime:
        pv = stl_preview(stl_preview_cached(stl), max_tris)
        THUMBS[stl] = (mtime, [tuple(round(c, 4) for c in v) for v in pv])   # 0.1 mm is plenty for a thumbnail
    return THUMBS[stl][1]


# ---------------------------------------------------------------- jobs
def run_cmd(cmd, cwd, capture=False, force=False, on_line=None):
    """The only place that spawns a subprocess. Streams output to the shared
    LOG; optionally also returns the command's own stdout for inspection
    (e.g. checkMesh's pass/fail message) without scraping the shared log.
    force=True runs even after Stop (cleanup that must leave the case usable).
    on_line(line) is called for every output line (solver progress). The running
    application's name goes to JOB["meta"]["step"] for the status pill."""
    if JOB["stop"] and not force:
        return (126, "") if capture else 126
    log(f"\n$ {cmd}\n")
    p = subprocess.Popen(foam_bash(cmd), cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    JOB["proc"] = p
    m = re.search(r"(\S+) -parallel", cmd)   # the application, also when wrapped in mpirun
    JOB["meta"]["step"] = m.group(1) if m else cmd.split()[0]
    chunks = []
    for line in p.stdout:
        log(line)
        if capture:
            chunks.append(line)
        if on_line:
            on_line(line)
    p.wait()
    if p.returncode != 0 and not JOB["stop"]:
        log(f"[command failed, exit code {p.returncode}]\n")
    return (p.returncode, "".join(chunks)) if capture else p.returncode


def par(cmd, n):
    """Wrap an OpenFOAM application for an MPI run over n ranks (n<=1: serial).
    Ranks are pinned one per physical core; more ranks than cores must be
    explicitly allowed (OpenMPI otherwise refuses with 'not enough slots')."""
    if n <= 1:
        return cmd
    bind = "--bind-to core" if n <= CORES else "--bind-to none --oversubscribe"
    return f"mpirun -np {n} {bind} {cmd} -parallel"


def decompose(case, n, extra=""):
    open(os.path.join(case, "system/decomposeParDict"), "w").write(DECOMPOSE % n)
    return run_cmd(f"decomposePar -force {extra}".strip(), case)


def run_one(item):
    kind, case = item["kind"], item["case"]
    JOB.update(running=True, stop=False, kind=kind, case=case, base=case, meta={})
    log(f"\n===== {kind} : {case} =====\n")
    ok = True
    try:
        item["fn"](*item["args"], **item["kwargs"])
    except Exception as e:
        ok = False
        log(f"\n[job error] {e}\n")
    ok = ok and not JOB["stop"]
    log("\n[finished]\n" if not JOB["stop"] else "\n[stopped]\n")
    JOB["seq"] += 1
    JOB["last_finish"] = {"kind": kind, "case": case, "ok": ok, "seq": JOB["seq"]}
    JOB["running"] = False


def queue_worker():
    while True:
        with QUEUE_LOCK:
            if not QUEUE:
                QUEUE_STATE["worker"] = False   # cleared under the lock: no lost wakeup
                return
            item = QUEUE.pop(0)
        run_one(item)


def run_job(kind, case, fn, *args, **kwargs):
    """Enqueue a job (mesh/solve/all/sweep); a single worker thread runs the
    queue FIFO. All subprocess/streaming logic lives in run_cmd(); run_one()
    owns JOB state. Returns the number of jobs ahead of this one."""
    with QUEUE_LOCK:
        QUEUE_STATE["next_id"] += 1
        QUEUE.append({"id": QUEUE_STATE["next_id"], "kind": kind, "case": case,
                      "fn": fn, "args": args, "kwargs": kwargs})
        ahead = len(QUEUE) - 1 + (1 if QUEUE_STATE["worker"] else 0)
        if not QUEUE_STATE["worker"]:
            QUEUE_STATE["worker"] = True
            threading.Thread(target=queue_worker, daemon=True).start()
    return ahead


def queue_snapshot():
    with QUEUE_LOCK:
        return [{"id": q["id"], "kind": q["kind"], "case": q["case"]} for q in QUEUE]


SKEW_TOLERATED = 20        # snappyHexMesh's own boundary-face skewness limit (meshQualityDict maxBoundarySkewness)
SKEW_FACE_FRACTION = 1e-4  # ... for at most this fraction of the faces (and at least 10 faces)


def mesh_check_text(text):
    """Classify checkMesh's own stdout -> (result, failed checks, warnings). OpenFOAM prints exactly one of
    'Mesh OK.' / 'Failed N mesh checks.' - see applications/utilities/mesh/manipulation/checkMesh.
    checkMesh flags any face with skewness > 4, but snappyHexMesh accepts up to 20 on the surface, so a
    snappy mesh often 'fails' on a handful of surface faces only. When skewness is the only failed check,
    for a few faces and within snappy's limit, the mesh counts as ok, with a warning."""
    if re.search(r"Mesh OK\.", text):
        return "ok", 0, []
    m = re.search(r"Failed (\d+) mesh checks\.", text)
    if not m:
        return "unknown", 0, []
    failed = int(m.group(1))
    skew = re.search(r"\*\*\*Max skewness = ([-+\d.eE]+), (\d+) highly skew faces", text)
    faces = re.search(r"^\s*faces:\s+(\d+)", text, re.M)
    if failed == 1 and skew and faces:
        worst, n = float(skew.group(1)), int(skew.group(2))
        if worst < SKEW_TOLERATED and n <= max(10, SKEW_FACE_FRACTION * int(faces.group(1))):
            return "ok", 0, [f"{n} face{'s' if n != 1 else ''} with skewness up to {worst:.3g}, above checkMesh's 4 "
                             f"but within snappyHexMesh's surface limit of {SKEW_TOLERATED}; tolerated"]
    return "failed", failed, []


_F = r"([-+\d.eE]+)"


def mesh_check_stats(text):
    """Mesh size and quality figures from checkMesh's own stdout. A check checkMesh flags with '***'
    (e.g. '***High aspect ratio cells found, Max aspect ratio: 1200') gets fail=True."""
    out = {}
    for key in ("cells", "faces", "points"):
        m = re.search(rf"^\s*{key}:\s+(\d+)", text, re.M)
        if m:
            out[key] = int(m.group(1))
    m = re.search(rf"non-orthogonality Max: {_F} average: {_F}", text)
    if m:
        out["non_orth"] = {"max": float(m.group(1)), "avg": float(m.group(2)),
                           "fail": "***Number of non-orthogonality errors" in text}
    m = re.search(rf"(\*\*\*)?Max skewness = {_F}", text)
    if m:
        out["skewness"] = {"max": float(m.group(2)), "fail": bool(m.group(1))}
    m = re.search(rf"(\*\*\*High aspect ratio cells found, )?Max aspect ratio[ =:]+{_F}", text)
    if m:
        out["aspect"] = {"max": float(m.group(2)), "fail": bool(m.group(1))}
    return out


def rewrite_snappy_level(case, level):
    info = read_gui_json(case)
    s = info.get("snappy") or {"loc": info.get("loc", [0, 0, 0])}   # pre-layers cases: no layers/wake
    open(os.path.join(case, "system/snappyHexMeshDict"), "w").write(snappy_dict(level, s))


def fix_side_patch_type(case):
    """Cases created before the fix wrote the 'slip' field BC as the sides *patch* type, which blockMesh
    accepts but foamRun rejects ('Unknown polyPatch type slip'). Repair blockMeshDict and the
    polyMesh boundary in place to a plain patch; the 0/ field BCs (type slip) stay as they are."""
    for rel in ("system/blockMeshDict", "constant/polyMesh/boundary"):
        path = os.path.join(case, rel)
        try:
            text = open(path).read()
        except OSError:
            continue
        fixed = re.sub(r"(\bsides\s*\{[^}]*?\btype\s+)slip\s*;", r"\1patch;", text)
        if fixed != text:
            open(path, "w").write(fixed)
            log(f"[repair] {rel}: sides patch type slip -> patch\n")


def read_mesh_status(case):
    try:
        return json.load(open(os.path.join(case, "mesh_status.json")))
    except OSError:
        return {}


def do_mesh(case, auto_retry=False, max_retries=2, level_bump=1, nprocs=1):
    level = read_gui_json(case).get("level", 3)
    fix_side_patch_type(case)
    history = []

    def write_status(attempt, lvl, result, failed=0, stats=None, warnings=()):
        history.append({"attempt": attempt, "level": lvl, "result": result, "failed_checks": failed})
        json.dump({"level": lvl, "attempts": attempt, "last_result": result, "last_failed_checks": failed,
                   "warnings": list(warnings), "stats": stats or {}, "history": history},
                  open(os.path.join(case, "mesh_status.json"), "w"))

    for c in ("surfaceCheck constant/geometry/part.stl", "blockMesh", "surfaceFeatures"):
        if JOB["stop"]:
            return
        if run_cmd(c, case) != 0:
            write_status(0, level, "pre_step_failed")
            return

    attempt = 0
    while True:
        attempt += 1
        JOB["meta"]["mesh_attempt"] = attempt
        if attempt > 1:
            level += level_bump
            log(f"\n[auto-retry] checkMesh failed; bumping refinement level to {level} and re-meshing (attempt {attempt})\n")
            rewrite_snappy_level(case, level)
            if JOB["stop"] or run_cmd("blockMesh", case) != 0:
                return
        if nprocs > 1:
            # 0/ references the 'body' patch snappy hasn't created yet: copy, don't decompose it
            if JOB["stop"] or decompose(case, nprocs, "-copyZero") != 0:
                return
        if JOB["stop"]:
            return
        rc = run_cmd(par("snappyHexMesh -overwrite", nprocs), case)
        if rc != 0:
            # record the failure: a stale 'ok' from an earlier mesh would let Mesh + Solve run on this one
            warnings = []
            if rc == 137 and not JOB["stop"]:   # 128 + SIGKILL: the kernel's OOM killer (also via mpirun)
                warnings = [f"snappyHexMesh ran out of memory at surface level {level}. maxGlobalCells is only "
                            "checked between refinement iterations, so the mesh can overshoot Max cells by ~1.5×. "
                            "Lower Max cells, use a coarser surface level/cell size, run fewer MPI processes, "
                            "or give WSL more memory (memory= in %UserProfile%\\.wslconfig)"]
                log(f"\n[mesh] {warnings[0]}\n")
            write_status(attempt, level, "out_of_memory" if warnings else "snappy_failed", warnings=warnings)
            if nprocs > 1:
                run_cmd("rm -rf processor*", case, force=True)   # leave a serial case behind
            return
        if nprocs > 1:
            # reconstructPar reconstructs the changed mesh; leave a serial case behind
            if JOB["stop"] or run_cmd("reconstructPar -constant -noFields && rm -rf processor*", case) != 0:
                return
        rc, text = run_cmd("checkMesh", case, capture=True)
        result, failed, warnings = mesh_check_text(text)
        for w_ in warnings:
            log(f"\n[mesh check] {w_}\n")
        write_status(attempt, level, result, failed, mesh_check_stats(text), warnings)
        if result == "ok" or not auto_retry or (attempt - 1) >= max_retries or JOB["stop"]:
            return


def watch_convergence(case, cfg, stop_event):
    """Short-lived thread, started just before the blocking foamRun call and
    joined right after - only calls log() and one subprocess for the graceful
    stop, never touches JOB fields, so there's no cross-thread write race."""
    tol = cfg.get("tol", 0.01)
    window = int(cfg.get("window", 20))
    need = int(cfg.get("consecutive", 3))
    min_iter = cfg.get("min_iter", 100)
    poll_s = cfg.get("poll_s", 5)
    consec = 0
    while not stop_event.wait(poll_s):
        data = parse_coeffs_series(case)
        cols, rows = data["columns"], data["rows"]
        if "Cd" not in cols or len(rows) < window:
            continue
        idx = cols.index("Cd")
        recent = rows[-window:]
        if recent[-1][0] < min_iter:
            continue
        cds = [r[idx] for r in recent]
        mean = sum(cds) / len(cds)
        rel = (max(cds) - min(cds)) / (abs(mean) or 1e-9)
        consec = consec + 1 if rel < tol else 0
        if consec >= need:
            log(f"\n[auto-stop] Cd converged (rel. spread {rel:.4g} < {tol} over last {window} samples) - requesting graceful stop\n")
            request_write_now(case)
            return


def request_write_now(case):
    """Gracefully stop a running (runTimeModifiable) solver: it writes the
    current time step and exits. Parallel ranks read the root controlDict too."""
    subprocess.run(foam_bash("foamDictionary system/controlDict -entry stopAt -set writeNow"),
                   cwd=case, capture_output=True, text=True)


def stop_job():
    """Stop the running job (the queue then continues). The first stop during
    foamRun lets it write the current fields and exit; any other, or a second, kills."""
    first = not JOB["stop"]
    JOB["stop"] = True
    if first and JOB["running"] and (JOB["meta"] or {}).get("phase") == "solve":
        log("\n[stop] asking the solver to write the current time step and exit (press Stop again to kill)\n")
        request_write_now(os.path.join(RUN, JOB["case"]))   # JOB["case"] is the name; the solver runs in its folder
    elif JOB["proc"] and JOB["proc"].poll() is None:
        os.killpg(os.getpgid(JOB["proc"].pid), signal.SIGTERM)


def restart_server(grace=120):
    """Drop the queue, stop the running job (gracefully first, killed after
    `grace` s), then re-exec this script. The listening socket isn't inherited
    and HTTPServer reuses the address, so the new process rebinds the port."""
    with QUEUE_LOCK:
        QUEUE.clear()
    if JOB["running"]:
        stop_job()
        deadline = time.time() + grace
        while JOB["running"] and time.time() < deadline:
            time.sleep(0.5)
        if JOB["running"]:
            stop_job()
            deadline = time.time() + 10
            while JOB["running"] and time.time() < deadline:
                time.sleep(0.5)
    log("\n[restart] restarting the server\n")
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


def do_solve(case, autostop=None, nprocs=1):
    if JOB["stop"] or run_cmd("rm -rf postProcessing processor* [1-9]*", case) != 0:
        return
    fix_side_patch_type(case)
    # a previous auto-stop leaves 'stopAt writeNow' in controlDict; reset it
    if run_cmd("foamDictionary system/controlDict -entry stopAt -set endTime > /dev/null", case) != 0:
        return
    info = read_gui_json(case)
    params = info.get("params") or {}
    transient = is_transient(params)
    init = (params.get("init_case") or "").strip() if transient else ""
    if init:
        # warm start: map the source case's latest solution onto this mesh as the initial fields (0/).
        # -consistent: both cases have the same domain and patches (e.g. one made from the other as a template)
        try:
            src = case_path(init)
        except ValueError as e:
            log(f"\n[warm start] {e}\n")
            return
        log(f"\n[warm start] initial fields mapped from the latest time of {init}\n")
        if run_cmd(f"mapFields {shlex.quote(src)} -consistent -sourceTime latestTime", case) != 0:
            return
    if nprocs > 1 and (JOB["stop"] or decompose(case, nprocs) != 0):
        return
    stop_event = threading.Event()
    watcher = None
    if autostop and autostop.get("enabled") and not transient:   # a transient run is meant to keep moving
        watcher = threading.Thread(target=watch_convergence, args=(case, autostop, stop_event), daemon=True)
        watcher.start()
    meta = JOB["meta"]
    # phase lets Stop request a graceful write instead of a kill; iter/iter0/end/t0 drive the progress bar
    # (iterations for a steady run, simulated seconds for a transient one: unit)
    end = info.get("end_time") or params.get("endtime")
    meta.update(phase="solve", iter=None, iter0=None, t0=time.time(), unit="s" if transient else "it",
                end=float(end or 0) or None)

    def progress(line):   # foamRun prints 'Time = 123s' at the start of every iteration
        m = re.match(r"Time = ([-+\d.eE]+)", line)
        if m:
            meta["iter"] = float(m.group(1))
            if meta["iter0"] is None:
                meta.update(iter0=meta["iter"], t0=time.time())

    run_cmd(par("foamRun", nprocs), case, on_line=progress)
    meta.pop("phase", None)
    stop_event.set()
    if watcher:
        watcher.join(timeout=2)
    if nprocs > 1:
        # even after Stop: whatever was written must end up in the serial case for ParaView. With no
        # time step written yet (stopped/failed before the first write) there's nothing to reconstruct:
        # reconstructPar skips 0/ and would fall back to 'constant' with a warning
        written = any(_num(t) for t in os.listdir(os.path.join(case, "processor0"))) \
            if os.path.isdir(os.path.join(case, "processor0")) else False
        run_cmd("reconstructPar && rm -rf processor*" if written else "rm -rf processor*", case, force=True)


def do_sample(case, lines, npts):
    """Sample U and p along `lines` (parse_probes() output) on the case's latest result, into
    postProcessing/lines/<latest time>/ - the same place the solve's own 'lines' function writes to - so
    lines can be added or changed after solving. Only this function object runs (not controlDict's, which
    would add a time folder to the force histories)."""
    # foamPostProcess -dict reads the file as the functions dictionary itself: entries at the top level
    open(os.path.join(case, "system/sampleLines"), "w").write(
        FOAMFILE % ("dictionary", "sampleLines") + lines_function(lines, npts))
    shutil.rmtree(os.path.join(case, "postProcessing/lines"), ignore_errors=True)
    run_cmd("foamPostProcess -dict system/sampleLines -latestTime", case)


def do_all(case, mesh_cfg=None, autostop_cfg=None, nprocs=1):
    do_mesh(case, nprocs=nprocs, **(mesh_cfg or {}))
    if JOB["stop"]:
        return
    if read_mesh_status(case).get("last_result") != "ok":
        log("\n[mesh+solve] mesh did not pass (or auto-retry was exhausted); skipping solve.\n")
        return
    do_solve(case, autostop_cfg, nprocs)


def sweep_row_summary(case, avg=AVG_WINDOW):
    """One comparison-table row: coefficients and forces averaged over the last avg iterations,
    plus the mesh's cell count (the x axis of a mesh-independence study)."""
    h = history(case, avg)
    out = {"cells": (read_mesh_status(case).get("stats") or {}).get("cells")}
    if h["coeffs"]["rows"]:
        out["Time"] = h["coeffs"]["rows"][-1][0]
    for group in ("coeffs", "forces"):
        out.update({c: a["mean"] for c, a in h["averages"][group].items()})
    return out


def do_sweep(base_params, param_name, values, mesh_cfg=None, autostop_cfg=None, nprocs=1, verts=None,
             avg=AVG_WINDOW):
    base_case = case_rel(base_params.get("case") or "gui_case")   # variants go next to it, in its folder
    table = []
    ref = None   # reference values and axes for the polar / yaw-stability summaries
    # angle (and, without y+ layer sizing, speed) sweeps don't change the mesh: mesh once, copy it after.
    # Not with symmetry sides: a tilted variant switches to freestream, a different patch type
    side = base_params.get("side_bc") or "slip"
    reuse = side != "symmetry" and (param_name in ("aoa", "yaw") or
                                    (param_name == "speed" and base_params.get("layer_mode") != "yplus"))
    mesh_src = None
    JOB["meta"].update(sweep_total=len(values), sweep_index=0, sweep_param=param_name)
    for i, v in enumerate(values):
        if JOB["stop"]:
            table.append({"variant": v, "status": "stopped"})
            break
        JOB["meta"].update(sweep_index=i + 1, sweep_value=v)
        p = dict(base_params)
        p[param_name] = v
        if param_name in ("cell_size_mm", "target_cells", "surface_cell_mm"):
            p["mesh_preset"] = "count" if param_name == "target_cells" else "surface"
        if param_name in ("aoa", "yaw") and v:
            p["side_bc"] = "freestream"   # a tilted inflow needs far-field sides
        p["case"] = f"{base_case}_sweep{i+1}"
        log(f"\n===== sweep {i+1}/{len(values)}: {param_name}={v} =====\n")
        try:
            info = setup_case(p, verts)
        except Exception as e:
            table.append({"variant": v, "case": p["case"], "status": "setup_failed", "error": str(e)})
            continue
        case = info["path"]
        JOB["case"] = info["case"]
        if mesh_src:
            log(f"[sweep] reusing the mesh of {os.path.basename(mesh_src)} (it doesn't change with {param_name})\n")
            shutil.rmtree(os.path.join(case, "constant/polyMesh"), ignore_errors=True)
            shutil.copytree(os.path.join(mesh_src, "constant/polyMesh"), os.path.join(case, "constant/polyMesh"))
            shutil.copy(os.path.join(mesh_src, "mesh_status.json"), case)
        else:
            do_mesh(case, nprocs=nprocs, **(mesh_cfg or {}))
        row = {"variant": v, "case": info["case"]}
        if JOB["stop"]:
            row["status"] = "stopped"
            table.append(row)
            break
        if read_mesh_status(case).get("last_result") != "ok":
            row["status"] = "mesh_failed"
            table.append(row)
            continue
        if reuse and not mesh_src:
            mesh_src = case
        if ref is None:
            ref = sweep_ref(case, base_params)
        do_solve(case, autostop_cfg, nprocs)
        row["status"] = "stopped" if JOB["stop"] else "done"
        row.update(sweep_row_summary(case, avg))
        table.append(row)
    LAST_SWEEP.clear()
    LAST_SWEEP.update(param=param_name, avg=avg, results=table, ref=ref)


def sweep_ref(case, base_params):
    """What the polar and yaw-stability summaries need from a sweep: reference values, density, speed,
    the axes and the base angle of attack (from the first meshed variant; equal for all of an angle sweep)."""
    g, info = case_geometry(case), read_gui_json(case)
    if not g:
        return None
    return {"S": info["aref"], "c": info["lref"], "b": g["span"], "rho": info["rho"], "speed": info["speed"],
            "axis": g["axis"], "sign": g["sign"], "lift_axis": g["lift_axis"], "pitch_axis": g["pitch_axis"],
            "cofr": g["cofr"], "aoa": float(case_params({"params": base_params}).get("aoa") or 0)}


def _fit(xs, ys):
    """Least-squares line y = a x + b -> (a, b); None for fewer than 2 points or no spread in x."""
    n = len(xs)
    if n < 2:
        return None
    mx_, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs)
    if sxx == 0:
        return None
    a = sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / sxx
    return a, my - a * mx_


def polar_summary(rows, ref, mass=None, target_sm=10):
    """Aircraft figures from an angle-of-attack sweep's mean coefficients (Cm positive nose-up, about the CG):
    lift slope, zero-lift angle, CLmax/stall, the parabolic drag polar CD = CD0 + k CL^2 with best L/D and
    Oswald efficiency, static margin -dCm/dCL with the neutral point and the CG for target_sm % margin, the
    trim angle (Cm = 0), and with mass (kg) the stall, cruise, best-glide and minimum-sink speeds."""
    pts = sorted((float(r["variant"]), r["Cl"], r["Cd"], r.get("Cm")) for r in rows
                 if r.get("status") == "done" and r.get("Cl") is not None and r.get("Cd") is not None)
    if len(pts) < 2 or not ref:
        return None
    a, cl, cd = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    imax = max(range(len(pts)), key=lambda i: cl[i])
    lin = slice(0, max(imax + 1, 2))   # the attached-flow part, up to CLmax
    out = {"points": [{"alpha": p[0], "CL": p[1], "CD": p[2], "Cm": p[3], "LD": p[1] / p[2] if p[2] else None}
                      for p in pts],
           "cl_max": cl[imax], "alpha_stall": a[imax], "stall_reached": imax < len(pts) - 1}
    f = _fit(a[lin], cl[lin])
    if f and f[0]:
        out.update(cl_alpha=f[0], alpha0=-f[1] / f[0])
    ld = [(cl[i] / cd[i], i) for i in range(len(pts)) if cd[i] > 0]
    if ld:
        best, i = max(ld)
        out.update(ld_max_pts=best, alpha_ld_max=a[i])
    f = _fit([c * c for c in cl[lin]], cd[lin])
    if f and f[0] > 0 and f[1] > 0:
        k, cd0 = f
        out.update(cd0=cd0, k=k, ld_max=1 / (2 * math.sqrt(cd0 * k)), cl_best_glide=math.sqrt(cd0 / k),
                   cl_min_sink=math.sqrt(3 * cd0 / k))
        if ref["b"] and ref["S"]:
            out["oswald"] = 1 / (math.pi * ref["b"] ** 2 / ref["S"] * k)
    cms = [p[3] for p in pts]
    if all(m is not None for m in cms):
        f = _fit(cl[lin], cms[lin])
        if f:
            sm = -f[0]   # static margin as a fraction of the reference chord
            ax, sgn, c = ref["axis"], ref["sign"], ref["c"]
            out.update(static_margin=sm, neutral_point=ref["cofr"][ax] + sgn * sm * c,
                       cg_target=ref["cofr"][ax] + sgn * (sm - target_sm / 100) * c,
                       target_sm=target_sm, flow_axis="xyz"[ax])
        f = _fit(a[lin], cms[lin])
        if f:
            out["cm_alpha"] = f[0]
        for i in range(len(pts) - 1):   # first sign change of Cm: trim
            if cms[i] == 0 or cms[i] * cms[i + 1] < 0:
                t = cms[i] / (cms[i] - cms[i + 1]) if cms[i] != cms[i + 1] else 0
                out.update(alpha_trim=a[i] + t * (a[i + 1] - a[i]), cl_trim=cl[i] + t * (cl[i + 1] - cl[i]))
                break
    if mass and mass > 0:
        v = lambda c_: flight_speed(mass, ref["rho"], ref["S"], c_)
        perf = {"mass": mass, "wing_loading": mass * G / ref["S"], "v_stall": v(out["cl_max"])}
        if out.get("cl_trim"):
            perf["v_trim"] = v(out["cl_trim"])
        if "cd0" in out and out["cl_best_glide"] <= out["cl_max"]:
            perf.update(v_best_glide=v(out["cl_best_glide"]), glide_ratio=out["ld_max"])
        if "cd0" in out and out["cl_min_sink"] <= out["cl_max"]:
            c_ = out["cl_min_sink"]
            vs = v(c_)
            perf.update(v_min_sink=vs, sink_rate=vs * (out["cd0"] + out["k"] * c_ * c_) / c_)
        out["perf"] = perf
    return out


def yaw_summary(rows, ref):
    """Lateral stability from a yaw (sideslip) sweep, in body axes x forward (upstream), z down (against
    lift), y = z x x (right wing): side force CY = Y/(qS), yawing Cn = N/(qSb), rolling Cl = L/(qSb) from the
    mean forces/moments about the CG; their slopes per radian of sideslip beta. Cn_beta > 0 is weathercock
    stable, Cl_beta < 0 a stable dihedral effect."""
    if not ref:
        return None
    ax, la, pa, sgn = ref["axis"], ref["lift_axis"], ref["pitch_axis"], ref["sign"]
    e = lambda i, s=1.0: [s if k == i else 0.0 for k in range(3)]
    xb, zb = e(ax, -sgn), e(la, -1.0)
    yb = [zb[(k + 1) % 3] * xb[(k + 2) % 3] - zb[(k + 2) % 3] * xb[(k + 1) % 3] for k in range(3)]
    dot = lambda u, w: sum(u[k] * w[k] for k in range(3))
    pts = []
    for r in rows:
        if r.get("status") != "done" or r.get("Fx") is None:
            continue
        al, be = math.radians(ref["aoa"]), math.radians(float(r["variant"]))
        flow = [math.cos(al) * math.cos(be) * e(ax, sgn)[k] + math.sin(al) * e(la)[k]
                + math.cos(al) * math.sin(be) * e(pa)[k] for k in range(3)]
        F, M = [r["Fx"], r["Fy"], r["Fz"]], [r["Mx"], r["My"], r["Mz"]]
        q = 0.5 * ref["rho"] * ref["speed"] ** 2
        qS, qSb = q * ref["S"], q * ref["S"] * ref["b"]
        beta = math.asin(max(-1.0, min(1.0, -dot(flow, yb))))   # wind from the right wing = positive
        pts.append({"beta": math.degrees(beta), "CY": dot(F, yb) / qS, "Cn": dot(M, zb) / qSb,
                    "Croll": dot(M, xb) / qSb})
    pts.sort(key=lambda p: p["beta"])
    if len(pts) < 2:
        return None
    out = {"points": pts}
    for key, name in (("Cn", "cn_beta"), ("Croll", "cl_beta"), ("CY", "cy_beta")):
        f = _fit([math.radians(p["beta"]) for p in pts], [p[key] for p in pts])
        if f:
            out[name] = f[0]
    return out


# ---------------------------------------------------------------- results / history parsing
def dimensionalize(colname, value, q, aref, lref):
    """OpenFOAM-13 forceCoeffs naming: a column starting 'Cm' is a moment
    coefficient, any other 'C...' column is a force coefficient. Generalizes to
    whichever subset of Cm/Cd/Cl/Cl(f)/Cl(r) is actually present."""
    if colname.startswith("Cm"):
        return value * q * aref * lref
    if colname.startswith("C"):
        return value * q * aref
    return None


def _sorted_time_subfolders(case, subdir, fname):
    files = glob.glob(os.path.join(case, subdir, "*", fname))

    def key(f):
        try:
            return float(os.path.basename(os.path.dirname(f)))
        except ValueError:
            return 0.0

    return sorted(files, key=key)


_DAT_TOKEN_RE = re.compile(r"\([^)]*\)|\S+")


def _num(tok):
    try:
        return float(tok)
    except ValueError:
        return None


def parse_dat_series(case, fo_name, fname):
    """Full time series from postProcessing/<fo_name>/*/<fname>, concatenated
    across all (numerically sorted) start-time subfolders. Columns come from
    the '# Time ...' header; a vector value '(x y z)' expands its column into
    name_x/name_y/name_z, and non-numeric tokens (e.g. solver names) are None."""
    columns, rows = [], []
    for fp in _sorted_time_subfolders(case, os.path.join("postProcessing", fo_name), fname):
        header = None
        for line in open(fp):
            if line.startswith("# Time"):
                header = line.lstrip("# ").split()
            elif not line.startswith("#") and line.strip() and header:
                toks = _DAT_TOKEN_RE.findall(line)
                if len(toks) != len(header):
                    continue
                cols, row = [], []
                for name, tok in zip(header, toks):
                    if tok.startswith("("):
                        comps = [_num(x) for x in tok.strip("()").split()]
                        cols += [f"{name}_{c}" for c in "xyz"[:len(comps)]]
                        row += comps
                    else:
                        cols.append(name)
                        row.append(_num(tok))
                if row[0] is None:
                    continue
                columns = cols
                rows.append(row)
    return {"columns": columns, "rows": rows}


def parse_coeffs_series(case):
    return parse_dat_series(case, "forceCoeffs", "forceCoeffs.dat")


PROBE_QTY = ("|U|", "Ux", "Uy", "Uz", "p")


def parse_probe_series(case):
    """The 'probes' function object's histories: postProcessing/probes/<t>/U and .../p, each '# Probe i (x y z)'
    header lines, a '# Time 0 1 ...' line, then one row per time step (U values as '(ux uy uz)'), concatenated
    over start-time folders. -> {names, t, q: {quantity: [one series per probe]}} on the U file's times."""
    def read(fname):
        rows, n = {}, 0
        for path in _sorted_time_subfolders(case, "postProcessing/probes", fname):
            for line in open(path):
                if line.startswith("#"):
                    n += line.startswith("# Probe")
                    continue
                toks = _DAT_TOKEN_RE.findall(line)
                t = _num(toks[0]) if toks else None
                if t is None or len(toks) < 2:
                    continue
                vals = [[_num(x) for x in tok.strip("()").split()] if tok.startswith("(") else _num(tok) for tok in toks[1:]]
                rows[t] = vals
        return rows, n
    u, n = read("U")
    p, _ = read("p")
    n = n // max(1, len(_sorted_time_subfolders(case, "postProcessing/probes", "U")))   # headers repeat per folder
    ts = sorted(t for t in u if len(u[t]) == n and all(isinstance(v, list) and len(v) == 3 for v in u[t]))
    q = {k: [[] for _ in range(n)] for k in PROBE_QTY}
    for t in ts:
        for i, v in enumerate(u[t]):
            q["|U|"][i].append(math.sqrt(sum(c * c for c in v)) if None not in v else None)
            for k, c in zip(("Ux", "Uy", "Uz"), v):
                q[k][i].append(c)
            pv = p.get(t)
            q["p"][i].append(pv[i] if pv and i < len(pv) and not isinstance(pv[i], list) else None)
    return {"names": [f"P{i + 1}" for i in range(n)], "t": ts, "q": q}


def parse_line_samples(case):
    """The 'lines' sets at the latest time they were written: postProcessing/lines/<t>/<name>.xy with columns
    'distance U_x U_y U_z p' -> {time, lines: [{name, d, q: {quantity: values}}]}, line1, line2... in order."""
    root = os.path.join(case, "postProcessing/lines")
    times = sorted((t for t in os.listdir(root) if _num(t) is not None), key=float) if os.path.isdir(root) else []
    if not times:
        return {"time": None, "lines": []}
    folder = os.path.join(root, times[-1])
    out = []
    for f in sorted((f for f in os.listdir(folder) if f.endswith(".xy")),
                    key=lambda f: int(re.sub(r"\D", "", f) or 0)):
        cols, rows = [], []
        for line in open(os.path.join(folder, f)):
            if line.startswith("#"):
                cols = line[1:].split()
                continue
            vals = [_num(x) for x in line.split()]
            if cols and len(vals) == len(cols) and None not in vals:
                rows.append(vals)
        if not rows or not {"distance", "U_x", "U_y", "U_z", "p"} <= set(cols):
            continue
        col = lambda name: [r[cols.index(name)] for r in rows]
        ux, uy, uz = col("U_x"), col("U_y"), col("U_z")
        out.append({"name": f[:-3], "d": col("distance"),
                    "q": {"|U|": [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(ux, uy, uz)],
                          "Ux": ux, "Uy": uy, "Uz": uz, "p": col("p")}})
    return {"time": _num(times[-1]), "lines": out}


_VEC_RE = re.compile(r"\(([^)]*)\)")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_force_series(case):
    """Full time series from postProcessing/forces/*/forces.dat (filename matches
    the function object's type name, same pattern as forceCoeffs.dat). Each line is
    'Time ((pressureForceVec) (viscousForceVec)) ((pressureMomentVec) (viscousMomentVec))'
    (OpenFOAM vector print format, no commas) - parsed with a regex, not .split(),
    then pressure+viscous summed elementwise for the total force/moment. The regex
    only matches innermost non-nested parens, so the extra outer paren pairs are
    simply skipped over (a stray leading '(' can end up inside a captured group,
    but the numeric regex applied to it ignores non-numeric characters)."""
    files = _sorted_time_subfolders(case, "postProcessing/forces", "forces.dat")
    # half model: report the full aircraft. Mirroring doubles the forces in the symmetry plane and cancels
    # the one across it; of the moments (about a CofR on the plane) only the one about the span axis remains
    half = read_gui_json(case).get("half_axis")
    fmul = mmul = [1, 1, 1]
    if half is not None:
        fmul = [0 if k == half else 2 for k in range(3)]
        mmul = [2 if k == half else 0 for k in range(3)]
    rows = []
    for fp in files:
        for line in open(fp):
            if line.startswith("#") or not line.strip():
                continue
            head = line.split(None, 1)
            if len(head) < 2:
                continue
            try:
                t = float(head[0])
            except ValueError:
                continue
            groups = _VEC_RE.findall(head[1])
            if len(groups) < 4:
                continue
            vecs = []
            for g in groups[:4]:
                nums = [float(x) for x in _NUM_RE.findall(g)]
                vecs.append((nums + [0.0, 0.0, 0.0])[:3])
            fp_, fv, mp, mv = vecs
            force = [(fp_[k] + fv[k]) * fmul[k] for k in range(3)]
            moment = [(mp[k] + mv[k]) * mmul[k] for k in range(3)]
            rows.append([t] + force + moment)
    return {"columns": ["Time", "Fx", "Fy", "Fz", "Mx", "My", "Mz"], "rows": rows}


def averages(series, window, t_window=None):
    """{column: {mean, min, max}} of a {columns, rows} series (Time excluded). Steady: over the last `window`
    rows - the converged value of a run that still oscillates a little, as SimScale reports it. Transient
    (t_window in s): time-weighted (trapezoidal) over the rows of the last t_window seconds, since the
    solver's adjustable time step spaces the rows unevenly."""
    rows = series["rows"]
    if t_window is not None:
        rows = [r for r in rows if r[0] >= rows[-1][0] - t_window] if rows else []
    else:
        rows = rows[-max(1, int(window)):]
    out = {}
    for i, name in enumerate(series["columns"][1:], start=1):
        pts = [(r[0], r[i]) for r in rows if i < len(r) and r[i] is not None]
        if not pts:
            continue
        vals = [v for _, v in pts]
        span = pts[-1][0] - pts[0][0]
        if t_window is not None and span > 0:
            mean = sum((t1 - t0) * (v0 + v1) / 2 for (t0, v0), (t1, v1) in zip(pts, pts[1:])) / span
        else:
            mean = sum(vals) / len(vals)
        out[name] = {"mean": mean, "min": min(vals), "max": max(vals)}
    return out


def is_transient(p):
    return (p or {}).get("sim_type") == "transient"


def avg_time_window(info, series, avg_time=None):
    """Seconds a transient case's coefficients are averaged over: avg_time if given, else the case's saved
    avg_time_s, else the last half of the simulated time (the start-up transient is in the first half)."""
    if not avg_time or avg_time <= 0:
        avg_time = opt_float(info.get("params") or {}, "avg_time_s")
    if not avg_time or avg_time <= 0:
        avg_time = series["rows"][-1][0] / 2 if series["rows"] else 0
    return avg_time


def case_averages(path, series, avg=AVG_WINDOW, avg_time=None, info=None):
    """averages() with the case's own kind of window: the last avg rows (steady) or seconds (transient)."""
    info = read_gui_json(path) if info is None else info
    if is_transient(info.get("params")):
        return averages(series, avg, avg_time_window(info, series, avg_time))
    return averages(series, avg)


def strouhal(ts, ys, lref, speed):
    """Dominant oscillation of a signal (e.g. a lift history) from its upward crossings of its own mean ->
    {f, St = f lref / U, cycles, amplitude} or None for fewer than 3 crossings (no steady oscillation yet).
    Crossings are interpolated in time, so the solver's uneven time steps don't bias the period."""
    if len(ts) < 4:
        return None
    mean = sum(ys) / len(ys)
    ups = [ts[i] + (ts[i + 1] - ts[i]) * (mean - ys[i]) / (ys[i + 1] - ys[i])
           for i in range(len(ys) - 1) if ys[i] < mean <= ys[i + 1]]
    if len(ups) < 3 or ups[-1] <= ups[0]:
        return None
    f = (len(ups) - 1) / (ups[-1] - ups[0])
    return {"f": f, "St": f * lref / speed if speed else None, "cycles": len(ups) - 1,
            "amplitude": (max(ys) - min(ys)) / 2}


def unsteady_summary(case, info, coeffs, forces, t_window):
    """Shedding figures of a transient case over its averaging window: the Strouhal number of the lift
    coefficient and of the side force (shedding shows in whichever force lies in the shedding plane)."""
    def window(series, col):
        if col not in series["columns"] or not series["rows"]:
            return [], []
        i, t1 = series["columns"].index(col), series["rows"][-1][0]
        pts = [(r[0], r[i]) for r in series["rows"] if r[0] >= t1 - t_window and r[i] is not None]
        return [p[0] for p in pts], [p[1] for p in pts]
    lref, speed = info.get("lref") or 0, info.get("speed") or 0
    out = {"lift": strouhal(*window(coeffs, "Cl"), lref, speed)}
    g = case_geometry(case)
    if g:
        out["side"] = strouhal(*window(forces, "F" + "xyz"[g["pitch_axis"]]), lref, speed)
        out["side_axis"] = "xyz"[g["pitch_axis"]]
    return out


def flight_speed(mass, rho, aref, cl):
    """Level-flight speed (m/s) at which lift coefficient cl carries mass (kg): L = m g."""
    return math.sqrt(2 * mass * G / (rho * aref * cl)) if mass and rho and aref and cl and cl > 0 else None


def history(case, avg=AVG_WINDOW, mass=None, avg_time=None):
    """mass (kg): the aircraft mass for the flight figures; None = the case's saved mass. avg: rows averaged
    over (steady); avg_time: seconds averaged over (transient; None = the case's default)."""
    info = read_gui_json(case)
    coeffs = parse_coeffs_series(case)
    forces = parse_force_series(case)
    dim_rows = []
    if coeffs["columns"] and info:
        q = 0.5 * info.get("rho", 0) * info.get("speed", 0) ** 2
        aref, lref = info.get("aref", 0), info.get("lref", 0)
        for row in coeffs["rows"]:
            drow = [row[0]]
            for name, val in zip(coeffs["columns"][1:], row[1:]):
                d = dimensionalize(name, val, q, aref, lref)
                drow.append(val if d is None else d)
            dim_rows.append(drow)
    res = parse_dat_series(case, "residuals", "residuals.dat")
    if res["rows"]:   # keep only numeric columns (drop e.g. solver-name columns)
        keep = [i for i, v in enumerate(res["rows"][-1]) if v is not None]
        res = {"columns": [res["columns"][i] for i in keep],
               "rows": [[r[i] for i in keep] for r in res["rows"] if len(r) == len(res["columns"])]}
    dim = {"columns": coeffs["columns"], "rows": dim_rows}
    tw = avg_time_window(info, coeffs, avg_time) if is_transient(info.get("params")) else None
    av_coeffs = averages(coeffs, avg, tw)
    if mass is None:
        mass = params_mass_kg(case_params(info))
    flight = None
    if mass and mass > 0 and info.get("aref"):
        cl = (av_coeffs.get("Cl") or {}).get("mean")
        flight = {"mass": mass, "wing_loading": mass * G / info["aref"],
                  "v_level": flight_speed(mass, info.get("rho"), info["aref"], cl)}
    return {
        "coeffs": coeffs,
        "forces": forces,
        "dimensional_coeffs": dim,
        "averages": {"window": min(int(avg), len(coeffs["rows"])), "window_s": tw, "coeffs": av_coeffs,
                     "dimensional": averages(dim, avg, tw), "forces": averages(forces, avg, tw)},
        "unsteady": unsteady_summary(case, info, coeffs, forces, tw) if tw else None,
        "transient": tw is not None,
        "flight": flight,
        "residuals": res,
        "inlet": parse_dat_series(case, "inletAvg", "surfaceFieldValue.dat"),
        "outlet": parse_dat_series(case, "outletFlow", "surfaceFieldValue.dat"),
        "probes": parse_probe_series(case),
        "lines": parse_line_samples(case),
        "info": info,
    }


def window_text(av):
    """What a history()'s averages are taken over, in words."""
    if av.get("window_s") is not None:
        return f"the last {av['window_s']:.4g} s (time-weighted)"
    return f"the last {av['window']} iterations"


def unsteady_text(u):
    """The shedding figures of history()['unsteady'] as text lines."""
    out = ""
    for key, label in (("lift", "lift coefficient Cl"), ("side", f"side force F{(u or {}).get('side_axis', '')}")):
        s = (u or {}).get(key)
        if s:
            out += (f"  Strouhal number from the {label}: St = {s['St']:.4g}  (f = {s['f']:.4g} Hz, "
                    f"{s['cycles']} cycles, amplitude {s['amplitude']:.4g})\n")
    return out


def results(case, avg=AVG_WINDOW, avg_time=None):
    h = history(case, avg, avg_time=avg_time)
    info, coeffs, forces, av = h["info"], h["coeffs"], h["forces"], h["averages"]
    if not coeffs["rows"]:
        return {"text": "No force results yet. Run the solver first."}
    last = coeffs["rows"][-1]
    txt = (f"{'Time' if h['transient'] else 'Iteration'}: {last[0]:.6g}{' s' if h['transient'] else ''}   "
           f"(values are means over {window_text(av)}, with their min..max range)\n")
    for i, name in enumerate(coeffs["columns"][1:], start=1):
        a = av["coeffs"].get(name)
        if not a:
            continue
        line = f"  {name} = {a['mean']:.5f}  [{a['min']:.5f} .. {a['max']:.5f}]   last {last[i]:.5f}"
        d = av["dimensional"].get(name)
        if d:
            line += f"   ({d['mean']:.5f} {'N m' if name.startswith('Cm') else 'N'})"
        txt += line + "\n"
    if forces["rows"]:
        f = {c: a["mean"] for c, a in av["forces"].items()}
        txt += "\nForces/moments (global frame, from the 'forces' function object, means):\n"
        txt += f"  Fx={f['Fx']:.5f} N  Fy={f['Fy']:.5f} N  Fz={f['Fz']:.5f} N\n"
        txt += f"  Mx={f['Mx']:.5f} N m  My={f['My']:.5f} N m  Mz={f['Mz']:.5f} N m\n"
    if h["transient"]:
        sh = unsteady_text(h["unsteady"])
        txt += "\nVortex shedding:\n" + (sh or "  no steady oscillation in the averaging window (fewer than 3 cycles)\n")
    inlet, outlet = h["inlet"], h["outlet"]
    if inlet["rows"] or outlet["rows"]:
        txt += "\nMonitors (kinematic p in m2/s2, as solved):\n"
    if inlet["rows"]:
        txt += "  inlet area-average: " + "  ".join(
            f"{c}={v:.5g}" for c, v in zip(inlet["columns"][1:], inlet["rows"][-1][1:])) + "\n"
    if outlet["rows"]:
        Q = outlet["rows"][-1][1]
        txt += f"  outlet flow rate Q={Q:.5g} m3/s   mass flow={Q * info.get('rho', 0):.5g} kg/s\n"
    if info:
        nu = info.get("nu", 1.5e-5)
        re_ = info.get("speed", 0) * info.get("lref", 0) / nu
        fl = h["flight"]
        if fl:
            txt += (f"\nAircraft mass {fl['mass']:g} kg: wing loading {fl['wing_loading']:.4g} N/m2"
                    + (f", level flight at this Cl needs {fl['v_level']:.3g} m/s" if fl["v_level"] else "") + "\n")
        cofr = info.get("cofr")
        txt += (f"\n(reference area {info.get('aref', 0):.6g} m2, length {info.get('lref', 0):.4g} m"
                + (", moment reference (" + ", ".join(f"{c:.4g}" for c in cofr) + ") m" if cofr else "")
                + f", rho {info.get('rho', 0)} kg/m3, nu {nu:g} m2/s, speed {info.get('speed', 0)} m/s, Re {re_:.4g}"
                + f", {TURB_MODELS.get(info.get('turb_model'), TURB_MODELS['SpalartAllmaras'])}"
                + (", half model mirrored to the full aircraft" if info.get("half_axis") is not None else "") + ")")
    return {"text": txt}


# ---------------------------------------------------------------- export: CSV history + HTML case report
def history_csv(case):
    """Every iteration's coefficients, their dimensional values and the forces/moments, joined on Time."""
    h = history(case)
    unit = lambda c: "N m" if c.startswith(("Cm", "M")) else "N"
    blocks = [(h["coeffs"], lambda c: c), (h["dimensional_coeffs"], lambda c: f"{c} [{unit(c)}]"),
              (h["forces"], lambda c: f"{c} [{unit(c)}]")]
    blocks = [(b, name) for b, name in blocks if b["rows"] and b["columns"]]
    header = ["Time"] + [name(c) for b, name in blocks for c in b["columns"][1:]]
    by_time = [{r[0]: r[1:] for r in b["rows"]} for b, _ in blocks]
    buf = io.StringIO()
    out = csv.writer(buf)
    out.writerow(header)
    for t in sorted({t for d in by_time for t in d}):
        row = []
        for (b, _), d in zip(blocks, by_time):
            row += d.get(t, [None] * (len(b["columns"]) - 1))
        out.writerow([f"{t:g}"] + ["" if v is None else f"{v:.8g}" for v in row])
    return buf.getvalue()


REPORT_COL = ("#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea", "#0891b2")   # light-background series


def svg_chart(series, title, log=False, w=640, h=220, xlabel="Iteration"):
    """A small static line chart (inline SVG) for the printable report: series = [(name, xs, ys)]."""
    pts = []
    for name, xs, ys in series:
        p = [(x, math.log10(y) if log else y) for x, y in zip(xs, ys)
             if y is not None and (y > 0 or not log)]
        pts.append((name, p))
    allp = [q for _, p in pts for q in p]
    if not allp:
        return ""
    L, R, T, B = 56, 12, 28, 30
    x0, x1 = min(q[0] for q in allp), max(q[0] for q in allp)
    fit = [q for q in allp if log or len(allp) <= 20 or q[0] >= x0 + 0.1 * (x1 - x0)]   # skip start-up spike
    y0, y1 = min(q[1] for q in fit), max(q[1] for q in fit)
    if log:
        y0, y1 = math.floor(y0), math.ceil(y1)
    y1, x1 = (y1 if y1 > y0 else y0 + 1), (x1 if x1 > x0 else x0 + 1)
    sx = lambda x: L + (w - L - R) * (x - x0) / (x1 - x0)
    sy = lambda y: T + (h - T - B) * (1 - (min(max(y, y0), y1) - y0) / (y1 - y0))
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(title)}">',
           f'<text x="{L}" y="16" class="t">{html.escape(title)}</text>']
    for k in range(5):
        v = y0 + (y1 - y0) * k / 4
        lab = f"1e{v:.0f}" if log and v == int(v) else ("" if log else f"{v:.4g}")
        out.append(f'<line x1="{L}" x2="{w - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="g"/>'
                   f'<text x="{L - 6}" y="{sy(v) + 4:.1f}" class="l" text-anchor="end">{lab}</text>')
    out.append(f'<text x="{(L + w - R) / 2}" y="{h - 8}" class="l" text-anchor="middle">{html.escape(xlabel)} '
               f'{x0:g} – {x1:g}</text>')
    for k, (name, p) in enumerate(pts):
        col = REPORT_COL[k % len(REPORT_COL)]
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.5" points="'
                   + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in p) + '"/>')
        if len(pts) > 1:
            out.append(f'<text x="{w - R - 70 * (len(pts) - k)}" y="16" class="l" fill="{col}">{html.escape(name)}</text>')
    return "".join(out) + "</svg>"


REPORT_CSS = """body{font:14px/1.45 system-ui,sans-serif;color:#1f2328;background:#fff;max-width:880px;margin:24px auto;padding:0 20px}
h1{font-size:20px;margin:0}h2{font-size:15px;margin:24px 0 8px;border-top:1px solid #d0d7de;padding-top:16px}
.m{color:#59636e;font-size:12.5px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
td,th{padding:4px 6px;border-bottom:1px solid #d0d7de;text-align:right}td:first-child,th:first-child{text-align:left}
th{color:#59636e;font-weight:500}pre{background:#f6f8fa;padding:12px;border-radius:8px;font-size:12px;white-space:pre-wrap}
svg .t{font:600 12px system-ui}svg .l{font:11px system-ui;fill:#59636e}svg .g{stroke:#d0d7de}.bad{color:#cf222e}
@media print{body{margin:0}h2{break-after:avoid}svg,table{break-inside:avoid}}"""


def report_html(case, avg=AVG_WINDOW, avg_time=None):
    """A standalone, printable HTML report of one case: settings, domain and mesh, averaged results and
    convergence charts (light theme for printing; not part of the dark UI)."""
    e = lambda v: html.escape(str(v))
    name = os.path.relpath(case, RUN)
    h = history(case, avg, avg_time=avg_time)
    xlabel = "Time (s)" if h["transient"] else "Iteration"
    info, av, ms, g = h["info"], h["averages"], read_mesh_status(case), case_geometry(case)
    params = case_params(info)   # the settings the case ran with
    table = lambda head, rows: ("<table><tr>" + "".join(f"<th>{e(x)}</th>" for x in head) + "</tr>" +
                                "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) +
                                "</table>")
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(name)} report</title>"
           f"<style>{REPORT_CSS}</style></head><body><h1>{e(name)}</h1>"
           f"<div class='m'>PIFOAM case report · OpenFOAM 13 · {e(time.strftime('%Y-%m-%d %H:%M'))}</div>"]
    out.append("<h2>Settings</h2>" + table(("Setting", "Value"), [(e(k), e(v)) for k, v in params.items()
                                                                    if str(v).strip() != ""]))
    if g:
        n = g["n"]
        rows = [("Reynolds number", f"{g['re']:.4g}"), ("Reference area A<sub>ref</sub>", f"{g['aref']:.4g} m²"),
                ("Reference length l<sub>ref</sub>", f"{g['lref']:.4g} m"),
                ("Moment reference (CG)", "(" + ", ".join(f"{c:.4g}" for c in g["cofr"]) + ") m"),
                ("Span", f"{g['span']:.4g} m" + (" (half model, mirrored)" if g["sym_face"] else "")),
                ("Turbulence model", e(TURB_MODELS.get(info.get("turb_model"), TURB_MODELS["SpalartAllmaras"]))),
                ("Flow direction", "(" + ", ".join(f"{c:.4f}" for c in g["flow"]) + ")"),
                ("Domain min / max (m)", ", ".join(f"{c:.3f}" for c in g["lo"]) + " / " +
                 ", ".join(f"{c:.3f}" for c in g["hi"])),
                ("Background cells", f"{n[0]} × {n[1]} × {n[2]} = {n[0] * n[1] * n[2]:,}"),
                ("Background / surface cell size", f"{g['h'] * 1000:.3g} mm / {g['surface_cell'] * 1000:.3g} mm")]
        if g["first_layer"]:
            rows.append((f"First layer for y+ {e(params.get('yplus') or 30)}", f"{g['first_layer'] * 1000:.3g} mm"))
        out.append("<h2>Domain</h2>" + table(("Quantity", "Value"), rows))
    if ms.get("last_result"):
        st = ms.get("stats") or {}
        rows = [("Result", e(ms["last_result"])), ("Refinement level / attempts", f"{ms['level']} / {ms['attempts']}")]
        rows += [(k.capitalize(), f"{st[k]:,}") for k in ("cells", "faces", "points") if k in st]
        for k, label in (("non_orth", "Max non-orthogonality (°)"), ("skewness", "Max skewness"),
                         ("aspect", "Max aspect ratio")):
            if k in st:
                rows.append((label, f"<span class='{'bad' if st[k]['fail'] else ''}'>{st[k]['max']:.4g}"
                                    f"{' (failed)' if st[k]['fail'] else ''}</span>"))
        out.append("<h2>Mesh</h2>" + table(("Quantity", "Value"), rows))
    if av["coeffs"]:
        rows = []
        for c, a in av["coeffs"].items():
            d = av["dimensional"].get(c)
            rows.append((e(c), f"{a['mean']:.5f}", f"{a['min']:.5f}", f"{a['max']:.5f}",
                         f"{d['mean']:.5g} {'N m' if c.startswith('Cm') else 'N'}" if d else ""))
        rows += [(e(c), f"{a['mean']:.5g}", f"{a['min']:.5g}", f"{a['max']:.5g}",
                  "N m" if c.startswith("M") else "N") for c, a in av["forces"].items()]
        out.append(f"<h2>Results</h2><div class='m'>Means over {e(window_text(av))}.</div>" +
                   table(("Quantity", "Mean", "Min", "Max", "Dimensional"), rows))
        cols, rws = h["coeffs"]["columns"], h["coeffs"]["rows"]
        for c, title in (("Cd", "Drag coefficient Cd"), ("Cl", "Lift coefficient Cl")):
            if c in cols:
                i = cols.index(c)
                out.append(svg_chart([(c, [r[0] for r in rws], [r[i] for r in rws])], title, xlabel=xlabel))
    res = h["residuals"]
    if res["rows"]:
        out.append(svg_chart([(c, [r[0] for r in res["rows"]], [r[k + 1] for r in res["rows"]])
                              for k, c in enumerate(res["columns"][1:])], "Initial residuals", log=True, xlabel=xlabel))
    out.append(f"<h2>Summary</h2><pre>{e(results(case, avg, avg_time)['text'])}</pre></body></html>")
    return "".join(out)


def case_template(name, keep_name=False):
    """A case's saved form parameters plus its (already metre-scaled) STL,
    loaded as the current geometry. keep_name=False is 'start from a study'
    (new name <case>_copy); keep_name=True is 'Load' (reopen the case itself,
    which also works for cases saved before settings were stored with them)."""
    case = case_path(name)
    base = case_rel(name)   # 'folder/case': a template's copy stays in the same folder
    params = read_gui_json(case).get("params")
    if not params and not keep_name:
        raise ValueError("This case was created before settings were saved with it - it can't be used as a template.")
    out = {"params": dict(params or {}, scale="1", case=base if keep_name else f"{base}_copy")}
    stl = os.path.join(case, "constant/geometry/part.stl")
    if os.path.isfile(stl):
        loaded = load_stl(open(stl, "rb").read(), f"{base}/{os.path.basename(stl)}", stl)
        verts = STATE["verts"]
        out.update(tris=loaded["tris"], ext=loaded["ext"], preview=loaded["preview"])
        p = out["params"]
        if p.get("mesh_preset") not in MESH_PRESETS and p.get("mesh_preset") != "surface":
            # an older resolution mode (legacy preset, cell size + level, cell count, cells across the part,
            # raw cell count) -> the same mesh as background + surface cell size in mm (the Custom mode)
            try:
                g = compute_geometry(verts, dict(case_params({"params": p}), max_cells="1e18"))   # converting only
                p.update(mesh_preset="surface", cell_size_mm=f"{g['h'] * 1000:.6g}",
                         surface_cell_mm=f"{g['h'] * 1000 / 2 ** g['level']:.6g}")
                for k in ("cells_per_part", "cells", "level", "target_cells"):
                    p.pop(k, None)
            except (ValueError, KeyError):
                pass   # incomplete old params: the form keeps its defaults
        if "mass" not in p and p.get("mass_kg"):   # mass used to be kg only
            p.update(mass=p.pop("mass_kg"), mass_unit="kg")
    return out


# ---------------------------------------------------------------- web page
# The UI's typefaces, embedded so the page needs no CDN: IBM Plex Sans Condensed and IBM Plex Mono, latin
# subsets from @fontsource 5.2.7. Copyright 2017 IBM Corp., SIL Open Font License 1.1 (scripts.sil.org/OFL).
# Injected into PAGE at the /*FONTS*/ placeholder.
_FONT_FILES = {
    ("IBM Plex Sans Condensed", 400): """
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
""",
    ("IBM Plex Sans Condensed", 500): """
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
AAA=
""",
    ("IBM Plex Sans Condensed", 600): """
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
LQHwlK76lot0sQaQfQEAAA==
""",
    ("IBM Plex Sans Condensed", 700): """
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
ogE=
""",
    ("IBM Plex Mono", 400): """
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
t6OiLwMAAAA=
""",
    ("IBM Plex Mono", 500): """
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
""",
}
FONTS = "".join("@font-face{font-family:'%s';font-weight:%d;font-style:normal;font-display:swap;"
                "src:url(data:font/woff2;base64,%s) format('woff2')}" % (fam, w, b64.replace("\n", ""))
                for (fam, w), b64 in _FONT_FILES.items())

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>PIFOAM</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
/*FONTS*/
/* PIFOAM design system: "precise engineering tool", dark. Everything visual is built from the tokens in :root
   and the primitives below; extend a primitive with a variant before adding a new one, and never add raw colours,
   sizes, radii, shadows or spacing outside :root.
   Primitives: layout (.grid .split .actions .spacer .scroll) · section .card (unboxed, top rule) + .sub ·
   surface .panel (.sunken, .interactive) · button (primary default, .sec, .ghost, .danger, .sm; :disabled) ·
   choice .seg / select / .tiles (no dropdown menus) · form field (label + control, .pair, label.check) ·
   .badge .pill .stats>.stat .readout .alert .note progress pre table .caserow · .titleblock (the current case,
   an engineering-drawing title block) · .dropzone .stlcanvas .chartcanvas · .steps · .tip · .ic.
   Data colours for canvases (--c1..--c6 series, --ax-* axes, --geo, --wake) are read by JS tok(). */
:root{
 color-scheme:dark;
 --bg:#0f0e0d;--bg-raised:#151413;--bg-elev:#171615;
 --card:rgba(255,250,240,.045);--card-hover:rgba(255,250,240,.075);
 --sub:#1b1a18;--fg:#ecebe7;--mut:#9b978f;
 --ac:#3463c9;--ac-hi:#3d6fd6;--on-ac:#fff;--bd:rgba(255,250,240,.07);--bd-hi:rgba(255,250,240,.13);--bd-ac:rgba(52,99,201,.35);
 --ok:#5fc48a;--bad:#ef7a6f;--warn:#e0b25a;--ac-bg:rgba(52,99,201,.12);--warn-bg:rgba(224,178,90,.1);--bad-bg:rgba(239,122,111,.1);
 --danger:#b8392c;--danger-hi:#c94535;--code-bg:#0b0a0a;--code-fg:#cfccc5;
 --c1:#7ea6e8;--c2:#e08a6d;--c3:#8fbf8a;--c4:#d4a95a;--c5:#6fb8b0;--c6:#c98fb0;
 --ax-x:#e0806a;--ax-y:#8cc08a;--ax-z:#7ea6e8;--geo:#9aa3ad;--wake:#d4a95a;
 --sh-lift:0 0 0 1px var(--bd-hi),0 4px 12px rgba(0,0,0,.3);
 --sh-btn:0 0 0 1px rgba(0,0,0,.35);
 --sh-sec:inset 0 0 0 1px var(--bd-hi);
 --grain:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 1 0 0 0 0 .98 0 0 0 0 .94 .07 0 0 0 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
 --font:"IBM Plex Sans Condensed","Arial Narrow",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
 --fs-2xs:11px;--fs-xs:12px;--fs-sm:13px;--fs-md:14px;--fs-lg:16px;--fs-xl:20px;--fs-2xl:28px;--fs-num:40px;
 --s0:2px;--s1:4px;--s2:8px;--s3:12px;--s4:16px;--s5:24px;--s6:32px;--s7:48px;--gutter:20px;--w-max:1080px;
 --r-ctl:4px;--r-surf:8px;--r-pill:999px;--bw:1px;--bw-2:2px;
 --ease:cubic-bezier(.16,1,.3,1);--t-fast:150ms;--t:220ms}
@media (max-width:600px){:root{--gutter:16px}}
@media (prefers-reduced-motion:reduce){*,*::before{transition:none!important;animation:none!important}}
*{box-sizing:border-box}html{background-color:var(--bg);background-image:var(--grain)}
body{margin:0;color:var(--fg);font:var(--fs-md)/1.45 var(--font);-webkit-font-smoothing:antialiased}
::selection{background:var(--bd-ac)}
/* ---- layout ---- */
.top{position:sticky;top:0;z-index:5;background:var(--bg-raised);border-bottom:var(--bw) solid var(--bd)}
.top .wrap{max-width:calc(var(--w-max) + 2*var(--gutter));margin:0 auto;padding:var(--s3) var(--gutter);display:flex;align-items:center;justify-content:space-between;gap:var(--s3)}
h1{font-size:var(--fs-xl);font-weight:700;letter-spacing:-.02em;margin:0;color:var(--fg)}h1 small{font:500 var(--fs-xs) var(--mono);color:var(--mut);letter-spacing:.04em;margin-left:var(--s2)}
.tag{font-size:var(--fs-sm);color:var(--mut)}
main{max-width:calc(var(--w-max) + 2*var(--gutter));margin:0 auto;padding:0 var(--gutter) var(--s7)}
.hright,.hleft{display:flex;align-items:center;gap:var(--s3)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:var(--s3)}
.dropzone+.grid,.grid+.grid,.grid+.tiles{margin-top:var(--s3)}
/* split: a workflow block's settings (left, one field row each) beside what they produce (right, sticky
   under the header while the inputs scroll); stacks under 1000px */
.split{display:grid;grid-template-columns:minmax(320px,440px) minmax(0,1fr);gap:var(--s3) var(--s7);align-items:start}
.split+.split{margin-top:var(--s5);padding-top:var(--s4);border-top:var(--bw) solid var(--bd)}
.split .effects{position:sticky;top:calc(var(--hdr,0px) + var(--s3))}
.split .effects>:first-child{margin-top:0}
@media (max-width:1000px){.split{grid-template-columns:minmax(0,1fr)}.split .effects{position:static}}
.split .inputs .grid{grid-template-columns:minmax(0,1fr);gap:var(--s2)}
.split .inputs .grid>div{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);align-items:center;gap:var(--s3)}
.split .inputs .grid>div>label{margin:0}.split .inputs label.check{grid-column:1/-1;padding-top:0}
.actions{display:flex;flex-wrap:wrap;align-items:center;gap:var(--s2);margin-top:var(--s3)}.actions.first{margin-top:0}.spacer{flex:1}
.scroll{overflow-x:auto}
/* ---- section card: a plain block, a thin rule and vertical rhythm, no box. Titles are large and condensed ---- */
.card{padding:var(--s6) 0 var(--s5);border-top:var(--bw) solid var(--bd)}main>.card:first-child{border-top:0;padding-top:var(--s4)}
.card-h{display:flex;align-items:baseline;flex-wrap:wrap;gap:var(--s3);margin-bottom:var(--s4)}
.card-h h2{font-size:var(--fs-2xl);line-height:1.1;margin:0;font-weight:600;letter-spacing:-.02em;color:var(--fg)}.card-h p{margin:0;font-size:var(--fs-sm);color:var(--mut)}
.card.locked>:not(.card-h):not(.lock-note){opacity:.4;pointer-events:none}.card:not(.locked) .lock-note{display:none}
/* collapsible card: the whole heading is the toggle */
details.card>summary.card-h{list-style:none;cursor:pointer;margin:0 0 var(--s3);font-size:inherit;color:inherit}
details.card>summary::-webkit-details-marker{display:none}
details.card:not([open])>summary.card-h{margin-bottom:0}
.chev{width:var(--s2);height:var(--s2);border-right:var(--bw-2) solid var(--mut);border-bottom:var(--bw-2) solid var(--mut);transform:rotate(-45deg);transition:transform var(--t) var(--ease);margin:0 var(--s1)}
details[open]>summary .chev{transform:rotate(45deg)}
.sub{font-size:var(--fs-lg);font-weight:600;color:var(--fg);letter-spacing:-.01em;margin:var(--s5) 0 var(--s3);padding-top:var(--s4);border-top:var(--bw) solid var(--bd)}
.sub.first{margin-top:0;padding-top:0;border-top:0}.sub.row{display:flex;align-items:center;flex-wrap:wrap;gap:var(--s2)}
/* ---- surface: the one contained block. .sunken = a recessed group (feature panel), .interactive = a raised
   item you act on (case/folder tiles, ParaView tiles) ---- */
.panel{border-radius:var(--r-surf);background:var(--card);color:var(--fg);box-shadow:var(--sh-sec)}
.panel.sunken{background:var(--sub);box-shadow:inset 0 0 0 var(--bw) var(--bd)}
.panel.interactive{transition:background-color var(--t-fast) var(--ease),box-shadow var(--t-fast) var(--ease),transform var(--t-fast) var(--ease)}
.panel.interactive:hover{background:var(--card-hover);box-shadow:var(--sh-lift);transform:translateY(-2px)}
.panel.interactive:active{transform:scale(.98)}
/* a valid drag-and-drop target: an inset accent ring (the accent marks the interaction) */
.panel.interactive.drop,.crumbs button.drop{box-shadow:inset 0 0 0 var(--bw-2) var(--ac),var(--sh-lift);background:var(--card-hover)}
/* home screen: the New case band, then the folder browser */
.newcase{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr) minmax(0,1.4fr) auto;gap:var(--s3);align-items:end}
@media (max-width:700px){.newcase{grid-template-columns:minmax(0,1fr)}}
input.search{width:200px;font-size:var(--fs-sm);padding:var(--s1) var(--s2)}
select.sm{width:auto;max-width:200px;font-size:var(--fs-sm);padding:var(--s1) var(--s2)}
#folderrow .dropzone.sm{flex:0 1 260px}
.crumbs{display:flex;flex-wrap:wrap;align-items:center;gap:var(--s0);font-size:var(--fs-lg);font-weight:600;margin-left:calc(-1*var(--s2))}
.crumbs button{padding:var(--s1) var(--s2);font:inherit}.crumbs button[aria-current]{color:var(--fg)}
.crumbs .sep{color:var(--mut);font-weight:400}
.casetile[draggable=true]{cursor:grab}
.thumb .ic.lg{display:block;width:var(--s7);height:var(--s7);margin:var(--s5) auto;color:var(--mut);stroke-width:1.5}
.casegrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:var(--s3)}
.casetile{display:flex;flex-direction:column;gap:var(--s2);padding:var(--s2) var(--s2) var(--s3)}
.casetile.panel.interactive:active{transform:none}
.casetile>:not(.thumb){margin:0 var(--s1)}
.casetile .name{font-size:var(--fs-lg);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.casetile .note{margin-top:0}.casetile .stats{margin-top:0}.casetile .actions{margin-top:auto}
button.thumb{display:block;padding:0;width:100%;background:var(--bg-elev);box-shadow:inset 0 0 0 var(--bw) var(--bd);border-radius:var(--r-ctl);overflow:hidden}
button.thumb:hover{background:var(--bg-elev)}
button.thumb canvas{display:block;width:100%;height:auto}
.thumb .note{display:block;padding:var(--s5) 0;text-align:center}
/* UI levels: Basic hides .mid-only and .adv-only, Intermediate (body.mid) hides .adv-only, Advanced
   (body.mid.adv) shows everything. Hidden fields keep their values and are still sent */
body:not(.mid) .mid-only,body:not(.adv) .adv-only{display:none!important}
/* per-case feature toggles: [data-feat] blocks of a switched-off feature (JS syncFeatures) */
.feat-off{display:none!important}
/* feature panel (.panel.sunken): a feature's fields, its On / Off .seg in the heading; Off collapses it */
.feature{margin-top:var(--s3);padding:var(--s3)}
.feature.first{margin-top:0}
.feature-h{display:flex;align-items:center;gap:var(--s2);font-size:var(--fs-md);font-weight:600;color:var(--fg)}
.feature-body{margin-top:var(--s3)}
.feature.off .feature-body{display:none}
.feature input:not([type=checkbox]),.feature select{background:var(--bg)}
/* ---- form fields ---- */
label{display:block;font-size:var(--fs-xs);color:var(--mut);margin-bottom:var(--s1)}
label.check{display:flex;align-items:center;gap:var(--s2);font-size:var(--fs-md);color:var(--fg);margin:0;padding-top:calc(var(--fs-xs)*1.45 + var(--s1));cursor:pointer}
label.check input{width:auto;margin:0;accent-color:var(--ac)}
.pair{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:var(--s2)}.pair select{width:auto}
input,select{width:100%;padding:calc(var(--s2) - var(--bw)) var(--s2);border:var(--bw) solid var(--bd-hi);border-radius:var(--r-ctl);background:var(--sub);color:var(--fg);font:inherit;font-variant-numeric:tabular-nums;transition:border-color var(--t-fast) var(--ease)}
input::placeholder{color:var(--mut);opacity:.7}
input:hover,select:hover{border-color:var(--mut)}
input:focus,select:focus{border-color:var(--ac);outline:var(--bw-2) solid var(--ac);outline-offset:-1px}
input:disabled,select:disabled{opacity:.45;cursor:not-allowed}
/* ---- button: primary by default; .sec (supporting), .ghost (tertiary), .danger, .sm. One primary per row ---- */
button{padding:var(--s2) var(--s4);border:0;border-radius:var(--r-ctl);background:var(--ac);color:var(--on-ac);font:inherit;font-weight:500;letter-spacing:.01em;cursor:pointer;box-shadow:var(--sh-btn);
 transition:background-color var(--t-fast) var(--ease),box-shadow var(--t-fast) var(--ease),color var(--t-fast) var(--ease),transform var(--t-fast) var(--ease)}
button:hover{background:var(--ac-hi)}button:active{transform:scale(.98)}
button:focus-visible{outline:var(--bw-2) solid var(--ac);outline-offset:var(--s0)}
button:disabled{opacity:.45;cursor:not-allowed;pointer-events:none}
button.sec{background:var(--card);color:var(--fg);box-shadow:var(--sh-sec)}button.sec:hover{background:var(--card-hover)}
/* ghost: no fill or ring until hovered. The segmented-control, breadcrumb and step buttons are ghosts too */
button.ghost,.seg button,.crumbs button,.steps button{background:transparent;box-shadow:none;color:var(--mut)}
button.ghost:hover,.seg button:hover,.crumbs button:hover,.steps button:hover{background:var(--card);color:var(--fg)}
.seg button.on,.steps button.on{background:var(--card-hover);color:var(--fg);box-shadow:var(--sh-sec)}
button.danger{background:var(--danger)}button.danger:hover{background:var(--danger-hi)}
button.sec.danger{background:var(--card);color:var(--bad)}button.sec.danger:hover{background:var(--card-hover)}
button.ghost.danger{background:transparent;color:var(--bad)}button.ghost.danger:hover{background:var(--bad-bg)}
button.sm{padding:var(--s1) var(--s3);font-size:var(--fs-sm)}
/* icon: inline SVG from JS icon(); stroke follows the text colour */
.ic{width:16px;height:16px;flex:none;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round;vertical-align:-3px}
.ic.sm{width:14px;height:14px;vertical-align:-2px}
button>.ic,.tiles>button>.ic{margin-right:var(--s2)}.badge>.ic{width:12px;height:12px;margin-right:var(--s1);vertical-align:-2px}
.steps small .ic,.note>.ic,.titleblock .ic{margin-right:var(--s1)}
/* ---- choice: .seg (pick one of a few), select (pick one of many), .tiles (actions with descriptions) ---- */
.seg{display:inline-flex;gap:var(--s0);padding:var(--s0);border-radius:var(--r-ctl);background:var(--sub);box-shadow:var(--sh-sec)}
.seg button{padding:var(--s1) var(--s3);border-radius:var(--r-ctl);font-size:var(--fs-sm)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:var(--s2)}
.tiles>button{text-align:left;padding:var(--s3)}
.tiles>button small{display:block;color:var(--mut);font-weight:400;font-size:var(--fs-xs)}
/* ---- step navigator: the workflow strip in the header, one button per workflow card ---- */
.steps{display:flex;gap:var(--s0);max-width:calc(var(--w-max) + 2*var(--gutter));margin:0 auto;padding:0 var(--gutter) var(--s2);overflow-x:auto;scrollbar-width:none}
.steps::-webkit-scrollbar{display:none}
.steps button{flex:1 0 auto;text-align:left;padding:var(--s1) var(--s3);font-size:var(--fs-sm)}
.steps .n{font:500 var(--fs-2xs) var(--mono);color:var(--mut);margin-right:var(--s1)}
.steps small{display:block;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:var(--fs-2xs);font-weight:400;color:var(--mut)}
/* ---- status & data display ---- */
.pill{display:inline-flex;align-items:center;gap:var(--s2);padding:var(--s1) var(--s3);border-radius:var(--r-pill);background:var(--card);box-shadow:var(--sh-sec);font-size:var(--fs-sm);color:var(--mut);white-space:nowrap}
.pill i{width:var(--s2);height:var(--s2);border-radius:50%;background:var(--mut)}.pill.run{color:var(--fg)}.pill.run i{background:var(--ac);animation:pulse 1.2s infinite}
@keyframes pulse{50%{opacity:.3}}
.badge{display:inline-block;font-size:var(--fs-2xs);padding:var(--s0) var(--s2);border-radius:var(--r-pill);border:var(--bw) solid var(--bd-hi);color:var(--mut)}
.badge.ok{color:var(--ok);border-color:currentColor}.badge.bad{color:var(--bad);border-color:currentColor}
.stats{display:flex;flex-wrap:wrap;gap:var(--s2);margin-top:var(--s3)}
.stat{background:var(--card);box-shadow:var(--sh-sec);border-radius:var(--r-ctl);padding:var(--s1) var(--s3);font-size:var(--fs-sm);font-variant-numeric:tabular-nums}
.stat b,.titleblock b,.readout b{display:block;font:500 var(--fs-2xs)/1.3 var(--mono);color:var(--mut);text-transform:uppercase;letter-spacing:.1em}
/* readout: a headline result figure (the averaged Cd / Cl / Cm), large mono numerals over a hairline */
.readouts{display:flex;flex-wrap:wrap;gap:var(--s3) var(--s6);padding-bottom:var(--s3);border-bottom:var(--bw) solid var(--bd)}
.readout b{text-transform:none;letter-spacing:.04em}
.readout .num{display:block;font:500 var(--fs-num)/1.1 var(--mono);letter-spacing:-.03em;font-variant-numeric:tabular-nums;color:var(--fg)}
.readout small{font-size:var(--fs-xs);color:var(--mut);font-variant-numeric:tabular-nums}
/* title block: the open case as an engineering-drawing title block - ruled cells, sharp corners, mono labels */
.titleblock{display:grid;grid-template-columns:minmax(0,2.4fr) repeat(5,minmax(0,1fr)) auto;border:var(--bw) solid var(--bd-hi)}
.titleblock>div{min-width:0;padding:var(--s2) var(--s3);border-left:var(--bw) solid var(--bd-hi)}
.titleblock>div:first-child{border-left:0}
.titleblock>div>span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:500 var(--fs-sm)/1.6 var(--mono);font-variant-numeric:tabular-nums}
.titleblock>.tb-name>span{font:600 var(--fs-xl)/1.2 var(--font);letter-spacing:-.01em}
.titleblock>.tb-act{display:flex;flex-direction:column;justify-content:center;align-items:stretch;gap:var(--s1)}
@media (max-width:900px){.titleblock{grid-template-columns:repeat(2,minmax(0,1fr))}
 .titleblock>div{border-left:0;border-top:var(--bw) solid var(--bd-hi)}.titleblock>div:first-child{border-top:0}
 .titleblock>div:nth-child(odd):not(:first-child):not(.tb-act){border-left:var(--bw) solid var(--bd-hi)}
 .titleblock>.tb-name,.titleblock>.tb-act{grid-column:1/-1}.titleblock>.tb-act{flex-direction:row;flex-wrap:wrap}}
.note{font-size:var(--fs-sm);color:var(--mut);margin-top:var(--s2)}.note.first,.actions .note{margin-top:0}
/* alert: tinted full-width message with a leading icon in its state colour (no side stripe) */
.alert{display:flex;flex-wrap:wrap;align-items:center;gap:var(--s2);margin-top:var(--s3);padding:var(--s2) var(--s3);border-radius:var(--r-ctl);box-shadow:inset 0 0 0 var(--bw) var(--bd);font-size:var(--fs-sm);color:var(--fg)}
.alert>span:not(.spacer){flex:1 1 200px}.alert>.ic{flex:none}
.alert.warn{background:var(--warn-bg)}.alert.warn>.ic{color:var(--warn)}
.alert.bad{background:var(--bad-bg)}.alert.bad>.ic{color:var(--bad)}
.alert.info{background:var(--ac-bg)}.alert.info>.ic{color:var(--ac-hi)}
.split .effects>.alert:first-child,.card-h+.alert{margin-top:0}
progress{display:block;width:100%;height:var(--s1);margin-top:var(--s3);-webkit-appearance:none;appearance:none;border:0;border-radius:0;background:var(--sub);box-shadow:var(--sh-sec);overflow:hidden}
progress::-webkit-progress-bar{background:var(--sub)}
progress::-webkit-progress-value{background:var(--ac)}progress::-moz-progress-bar{background:var(--ac)}
pre{background:var(--code-bg);color:var(--code-fg);padding:var(--s3);border-radius:var(--r-surf);box-shadow:0 0 0 var(--bw) var(--bd);height:240px;overflow:auto;font:var(--fs-xs)/1.45 var(--mono);margin:var(--s2) 0 0;white-space:pre-wrap}
pre.results{height:auto;min-height:40px;background:var(--sub);color:var(--fg)}
table{width:100%;border-collapse:collapse;font-size:var(--fs-xs);margin-top:var(--s2);font-variant-numeric:tabular-nums}
th,td{padding:var(--s1) var(--s2);text-align:right;border-bottom:var(--bw) solid var(--bd);white-space:nowrap}
th{color:var(--mut);font:500 var(--fs-2xs) var(--mono);text-transform:uppercase;letter-spacing:.06em}th:first-child,td:first-child{text-align:left}
.caserow{display:flex;align-items:center;gap:var(--s2);padding:var(--s2) 0;border-bottom:var(--bw) solid var(--bd);flex-wrap:wrap}
.caserow:last-child{border-bottom:0}.caserow .name{font-weight:600;margin-right:var(--s1)}
summary{cursor:pointer;font-size:var(--fs-sm);color:var(--mut);margin-top:var(--s3)}
/* ---- surfaces: drop zone, 3D viewer, chart ---- */
.dropzone{border:var(--bw) dashed var(--bd-hi);border-radius:var(--r-surf);padding:var(--s4);text-align:center;background:var(--sub);transition:border-color var(--t-fast) var(--ease),background-color var(--t-fast) var(--ease)}
.dropzone.sm{padding:var(--s1) var(--s2);text-align:left;border-radius:var(--r-ctl)}.dropzone.sm input{padding:var(--s1) 0;font-size:var(--fs-sm)}
.dropzone.has-file{border-style:solid}.dropzone.drag{border-color:var(--ac);background:var(--card)}.dropzone input{width:auto;background:none;border:0}
.stlcanvas,.chartcanvas{width:100%;height:auto;background:var(--bg-elev);border:var(--bw) solid var(--bd);border-radius:var(--r-surf);display:block;margin-top:var(--s2)}
.stlcanvas{touch-action:none;cursor:grab}
/* ---- hover help: one floating .tip, filled from the hovered/focused element's data-tip ---- */
.tip{position:fixed;left:0;top:0;z-index:10;max-width:320px;padding:var(--s2) var(--s3);border-radius:var(--r-ctl);background:var(--bg-elev);box-shadow:var(--sh-lift);
 color:var(--fg);font-size:var(--fs-xs);line-height:1.45;pointer-events:none;opacity:0;transform:translateY(-2px);transition:opacity var(--t-fast) var(--ease),transform var(--t-fast) var(--ease)}
.tip.on{opacity:1;transform:none}
[data-tip]>label:not(.check){cursor:help}
</style></head><body>
<header class="top" id="top"><div class="wrap">
 <div><div class="hleft"><h1>PIFOAM<small>OpenFOAM 13</small></h1><button class="ghost sm" id="restartbtn" onclick="restartServer()" data-tip="Restarts the PIFOAM server, e.g. to pick up code changes. A running job is stopped first (a solver writes its current fields), queued jobs are dropped, and the page reloads once the server is back.">Restart</button></div><div class="tag" id="tagline">Open a recent case or start a new one.</div></div>
 <div class="hright">
  <div id="uimode" class="seg"><button data-m="basic" class="on" data-tip="Only the essentials: geometry, flow, mesh resolution, then mesh and solve. Hidden settings keep their values and defaults.">Basic</button><button data-m="mid" data-tip="Adds boundary layers, wake refinement, domain size, mesh-retry and auto-stop tuning, and parameter sweeps.">Intermediate</button><button data-m="adv" data-tip="Everything: also inflow turbulence, side-wall type, snappyHexMesh controls and solver numerics.">Advanced</button></div>
  <span id="status" class="pill"><i></i><span id="statustext">Idle</span></span>
 </div>
</div><nav id="steps" class="steps" aria-label="Workflow steps" style="display:none"></nav></header>
<main>

<section class="card" id="casecard"><div class="card-h"><h2>Case</h2><p id="casehint">Open a recent case or start a new one</p></div>
<div id="casepick">
 <div class="sub first">New case</div>
 <div class="newcase">
  <div data-tip="The case's own directory name, inside the folder picked next to it. Letters, digits and underscores; anything else becomes _. Choosing an STL fills it from the file name while you haven't typed one."><label>Case name</label><input id="case" value="gui_case"></div>
  <div data-tip="The folder the case goes in; it follows the folder you have open below. A folder with a shared part (its own or an enclosing folder's) gives new cases that part when you don't choose a file."><label>Folder</label><select id="newfolder"><option value="">Top level</option></select></div>
  <div data-tip="What the case simulates. Steady external flow converges to one flow field; transient external flow follows it in time (vortex shedding, unsteady wakes)."><label>Simulation type</label><select id="sim_type"></select></div>
  <div data-tip="Optional: the part's STL, uploaded as soon as the case opens. Without one, a case in a folder with a shared part starts from that part. You can also add or replace it later in the Geometry step."><label>Part (STL)</label>
   <div id="newdrop" class="dropzone sm"><input type="file" id="newfile" accept=".stl"></div></div>
  <button data-ic="file" onclick="startCase()" data-tip="Start working on a new case with this name. Nothing is written to disk until you click Create case files.">Create</button>
 </div>
 <div class="sub row"><nav id="crumbs" class="crumbs" aria-label="Folder"></nav><span class="spacer"></span>
  <input id="casesearch" class="search" type="search" placeholder="Search all cases" aria-label="Search all cases" data-tip="Show the cases whose name or folder contains this text, from every folder.">
  <div id="casesort" class="seg"><button class="on" data-s="recent" data-tip="Most recently changed first.">Recent</button><button data-s="name" data-tip="Alphabetical.">Name</button></div>
  <button class="sec sm" onclick="newFolder()" data-tip="Create a folder inside the one you have open, to group related cases (and folders). It can hold a shared part for its new cases.">New folder</button>
  <button class="sec sm" onclick="loadCases()" data-tip="Scan $FOAM_RUN again for cases and folders.">Refresh</button></div>
 <div id="folderrow" class="caserow" style="display:none"></div>
 <div id="caselist" class="note">Loading...</div>
</div>
<div id="curcase" class="titleblock" style="display:none"></div>
</section>

<div id="workspace" style="display:none">
<section class="card" id="geomcard"><div class="card-h"><h2>Geometry</h2><p>The part the flow goes around</p></div>
<div id="dropzone" class="dropzone" data-tip="The part's surface as a binary or ASCII STL. It's what gets meshed around; the flow domain is sized from its bounding box.">
  <input type="file" id="file" accept=".stl">
  <div id="bbox" class="note" data-ic="upload">No file chosen. Pick an STL or drag it here.</div>
</div>
<div class="grid">
 <div data-tip="The units the STL was exported in; coordinates are converted to metres. Guessed from the part size on upload, so check it."><label>STL units</label><select id="scale"><option value="0.001">millimetres</option><option value="1">metres</option><option value="0.01">centimetres</option><option value="0.0254">inches</option></select></div>
</div>
<div class="feature panel sunken mid-only" data-feature="half"><div class="feature-h"></div><div class="feature-body">
<div class="grid">
 <div data-tip="Simulate half of a symmetric part to save cells and time: a symmetry plane cuts the whole part across the span axis (neither the flow nor the lift axis) and only the side above it is meshed and solved. Forces, moments and coefficients are reported for the whole part, and ParaView mirrors the result so every view shows all of it. Not combinable with yaw. Off simulates the full model."><label>Cut plane</label><select id="half_model"><option value="none" selected>Off (full model)</option><option value="centre">Centre of the part</option><option value="custom">Position (mm)</option><option value="min" hidden>Part is one half, root at the span-axis minimum</option><option value="max" hidden>Part is one half, root at the span-axis maximum</option></select></div>
 <div data-when="half_model=custom" data-tip="The cut plane's coordinate on the span axis, in mm in the part's own coordinates (after the STL unit conversion). Use it when the part's plane of symmetry isn't the middle of its bounding box."><label>Cut plane position (mm)</label><input id="half_mm" type="number" step="any" placeholder="required"></div>
</div>
</div></div>
<input type="hidden" id="features" value="">
</section>

<section class="card" id="physcard"><div class="card-h"><h2>Physics</h2><p>Fluid, inflow and boundaries</p></div>
<div class="split"><div class="inputs">
<div class="sub first">Fluid and inflow</div>
<div class="grid">
 <div data-tip="Pick a preset fluid to fill in density and viscosity. Editing either value switches this to Custom fluid."><label>Fluid</label><select id="fluid"></select></div>
 <div data-tip="Free-stream velocity at the inlet. Sets the Reynolds number and the dynamic pressure that turns coefficients into forces."><label>Flow speed (m/s)</label><input id="speed" type="number" value="15"></div>
 <div data-tip="The axis and sense the flow travels along. The inlet is the domain face upstream of the part, the outlet the face downstream; the arrow in the viewer shows it."><label>Flow direction</label><select id="dir"><option>+x</option><option selected>+y</option><option>+z</option><option>-x</option><option>-y</option><option>-z</option></select></div>
 <div data-tip="Which axis points up in your STL. Lift, the angle of attack, the ground and Cm are measured against it, and the remaining axis is the span (wingspan). CAD tools differ: many export Z up, Fusion 360 and some others Y up. Automatic: Z up, or Y up when the flow runs along Z."><label>Up axis</label><select id="up_axis"><option value="auto" selected>Automatic (Z, or Y)</option><option value="z">Z up</option><option value="y">Y up</option><option value="x">X up</option></select></div>
 <div data-tip="Only used to convert coefficients into forces (N) and moments (Nm). The solver itself is incompressible and works with pressure divided by density."><label>Density (kg/m3)</label><input id="rho" type="number" value="1.204" step="0.1"></div>
 <div data-tip="ν = dynamic viscosity ÷ density. Sets the Reynolds number (speed × part length ÷ ν) and the inlet turbulence level."><label>Kinematic viscosity (m2/s)</label><input id="nu" type="number" value="1.5e-5" step="any"></div>
</div>
<div class="feature panel sunken mid-only" data-feature="angles"><div class="feature-h"></div><div class="feature-body"><div class="grid">
 <div data-tip="Tilts the inflow toward the up axis, like pitching a wing up. Lift and drag are measured perpendicular and parallel to the tilted flow. Needs Freestream side walls, which are switched on for you. Up to ±30°."><label>Angle of attack (&deg;)</label><input id="aoa" type="number" value="0" step="1" min="-30" max="30"></div>
 <div data-tip="Tilts the inflow sideways, toward the remaining axis, like a crosswind on a car. Needs Freestream side walls, which are switched on for you. Up to ±30°."><label>Yaw (&deg;)</label><input id="yaw" type="number" value="0" step="1" min="-30" max="30"></div>
</div></div></div>
<div class="mid-only">
<div class="feature panel sunken" data-feature="ground"><div class="feature-h"></div><div class="feature-body"><div class="grid">
 <div data-tip="Turns the bottom of the domain (the face below the part along the lift axis) into a wind-tunnel floor at the part's lowest point. Fixed: a still floor. Moving: a belt running at the flow speed, as for a car on the road. Not combinable with an angle of attack."><label>Ground</label><select id="ground"><option value="none" selected>None (free stream)</option><option value="fixed">Fixed floor</option><option value="moving">Moving ground</option></select></div>
 <div class="adv-only" data-tip="Gap between the floor and the part's lowest point, in real millimetres. 0 puts the floor right at the part; a small negative value sinks wheels slightly into the floor, which meshes more reliably than an exact touch."><label>Ground clearance (mm)</label><input id="ground_gap_mm" type="number" value="0" step="any"></div>
</div></div></div>
<div class="feature panel sunken" data-feature="aircraft"><div class="feature-h"></div><div class="feature-body">
<div class="grid">
 <div data-tip="What the coefficients are scaled by. Automatic: the part's frontal area, its length along the flow and its centre (fine for cars and bodies). Aircraft: enter the wing area, the mean aerodynamic chord and the CG, so Cl, Cd and Cm compare with other aircraft and the stability figures refer to your CG."><label>Reference values</label><select id="ref_mode"><option value="auto" selected>Automatic (frontal area, length)</option><option value="custom">Aircraft (wing area, chord, CG)</option></select></div>
 <div data-when="ref_mode=custom" data-tip="Wing planform area of the whole aircraft, both wings, in m², in the units after the STL conversion. With a half model still enter the full area."><label>Wing area (m&sup2;)</label><input id="aref" type="number" step="any" min="0"></div>
 <div data-when="ref_mode=custom" data-tip="Mean aerodynamic chord in metres. It scales Cm and the Reynolds number, and the static margin is given as a fraction of it."><label>Mean aerodynamic chord (m)</label><input id="lref" type="number" step="any" min="0"></div>
 <div data-when="ref_mode=custom" data-tip="X coordinate of the centre of gravity, in metres in the part's own coordinates (after the STL unit conversion). Moments are taken about this point. Blank uses the part's centre."><label>CG X (m)</label><input id="cofr_x" type="number" step="any"></div>
 <div data-when="ref_mode=custom" data-tip="Y coordinate of the centre of gravity, in metres. Blank uses the part's centre. With a half model the CG is put on the symmetry plane."><label>CG Y (m)</label><input id="cofr_y" type="number" step="any"></div>
 <div data-when="ref_mode=custom" data-tip="Z coordinate of the centre of gravity, in metres. Blank uses the part's centre."><label>CG Z (m)</label><input id="cofr_z" type="number" step="any"></div>
 <div data-tip="The aircraft's total flying mass, battery included. It isn't used by the simulation, only to turn the results into wing loading and flight speeds, so you can change it at any time without re-running. Blank = no flight figures."><label>Aircraft mass</label><div class="pair"><input id="mass" type="number" step="any" min="0" placeholder="off" aria-label="Aircraft mass"><select id="mass_unit" aria-label="Mass unit"><option value="g" selected>g</option><option value="kg">kg</option></select></div></div>
</div>
<div class="note">Selecting Aircraft starts from the automatic values, so replace them with your wing's.</div>
</div></div>
<div class="sub">Turbulence and boundaries</div>
<div class="grid">
 <div data-tip="Spalart–Allmaras: robust and cheap, fine for attached flow. k-ω SST: better near separation and stall. k-ω SST with transition: also predicts where the boundary layer turns from laminar to turbulent, which matters at RC-model Reynolds numbers (laminar bubbles, lower drag), but needs a near-wall mesh with y+ about 1."><label>Turbulence model</label><select id="turb_model"></select></div>
 <div class="adv-only" data-tip="Inflow turbulence: the turbulent viscosity at the inlet as a multiple of ν (Spalart–Allmaras: ν̃/ν; k-ω models: ν_t/ν, which sets ω). 3 is a typical free-stream value; higher means a more turbulent inflow."><label>Inlet turbulent viscosity ratio</label><input id="nut_ratio" type="number" value="3" step="0.5" min="0"></div>
 <div class="adv-only" data-when="turb_model=kOmegaSST kOmegaSSTLM" data-tip="Turbulence intensity of the incoming air, in percent: velocity fluctuations relative to the flow speed. Calm free flight is about 0.1 %, a wind tunnel 0.2–1 %. With the transition model it also sets where transition starts."><label>Inlet turbulence intensity (%)</label><input id="turb_intensity" type="number" value="0.1" step="0.05" min="0"></div>
 <div class="adv-only" data-when="turb_model=kOmegaSST kOmegaSSTLM" data-tip="How the inlet ω is set. Viscosity ratio: from k and the turbulent viscosity ratio above. Length scale: from k and the size of the incoming eddies, ω = √k / (Cμ^¼ L) — use it when you know the inflow's turbulence scale (e.g. a wind tunnel's grid)."><label>Inlet ω from</label><select id="omega_mode"><option value="ratio">Viscosity ratio</option><option value="length">Length scale</option></select></div>
 <div class="adv-only" data-when="turb_model=kOmegaSST kOmegaSSTLM;omega_mode=length" data-tip="Size of the energy-carrying eddies in the inflow, in mm. Roughly 7 % of a duct's hydraulic diameter, or a few times a wind-tunnel grid's mesh size; free flight is much larger."><label>Inlet turbulence length scale (mm)</label><input id="turb_length_mm" type="number" step="any" min="0" placeholder="required"></div>
 <div class="adv-only" data-tip="Condition on the side faces of the domain. Slip wall: flow slides along without friction. Symmetry plane: a mirror. Freestream: far field, flow may enter or leave, required for a tilted inflow. None of them grows a boundary layer."><label>Side walls</label><select id="side_bc"><option value="slip" selected>Slip wall</option><option value="symmetry">Symmetry plane</option><option value="freestream">Freestream (far field)</option></select></div>
</div>
</div>
</div><div class="effects">
<div class="actions first"><div id="physmode"></div><span class="spacer"></span><div id="physviews"></div></div>
<canvas id="physview" width="720" height="380" class="stlcanvas"></canvas>
<div class="note">The arrow is the inflow, tilted by the angle of attack and yaw. The ground and symmetry planes show when they're on, and the quartered CG roundel marks the moment reference point.</div>
<div id="physstats" class="note">Load an STL to see the flow figures.</div>
<div id="flightstats" class="mid-only" data-feat="aircraft"></div>
<div id="flightnote" class="note mid-only" data-feat="aircraft"></div>
<div id="physwarn" style="display:none"></div>
<div class="note mid-only" data-feat="aircraft">Cm is positive nose-up, about the CG.</div>
<div class="note" data-when="turb_model=kOmegaSSTLM">The transition model resolves the boundary layer down to the wall: under Mesh, turn on boundary layers with Layer sizing set to First layer from y+ and Target y+ 1. Meshing and solving take noticeably longer.</div>
</div></div>
</section>

<section class="card" id="meshcard"><div class="card-h"><h2>Mesh</h2><p>Resolution, boundary layers, refinement and domain</p></div>
<div class="split"><div class="inputs">
<div class="sub first">Resolution</div>
<div class="grid">
 <div data-tip="A starting point for the two cell sizes below: the surface cell size is fixed (Coarse 4 mm, Medium 2 mm, Fine 1 mm, Very fine 0.5 mm) and the background cell is sized so the domain holds about 150k / 250k / 400k / 600k background cells. Editing either size switches this to Custom."><label>Resolution</label><select id="mesh_preset"><option value="surf4">Coarse (4 mm surface)</option><option value="surf2" selected>Medium (2 mm surface)</option><option value="surf1">Fine (1 mm surface)</option><option value="surf05">Very fine (0.5 mm surface)</option><option value="surface">Custom</option></select></div>
 <div data-tip="Edge length of the background mesh cells in real millimetres, whatever the STL units. It sets how fine the flow is resolved away from the part. Halving it gives about 8× as many background cells. The count shows on the right and must stay under Max cells."><label>Background cell size (mm)</label><input id="cell_size_mm" type="number" value="50" step="any" min="0"></div>
 <div data-tip="The cell size you want on the part's surface, in real millimetres. The refinement level is the fewest halvings of the background cell that reach it, so the actual surface cells (on the right) are this size or a little finer. Smaller resolves curves, thin edges and the boundary layer better, at many more cells. For a wing, aim for 50–100 cells across the chord."><label>Surface cell size (mm)</label><input id="surface_cell_mm" type="number" value="2" step="any" min="0"></div>
</div>
<div class="note">A preset fills both sizes from the part and domain; change either one to fine-tune it (Custom). The surface refinement level follows from the two.</div>
</div><div class="effects">
<div class="sub first">Cell sizes</div>
<div id="cellinfo" class="note">Load an STL to see the background and refined cell sizes.</div>
<div id="cellnote" class="note"></div>
</div></div>
<div class="split mid-only"><div class="inputs">
<div class="sub first">Boundary layers</div>
<div class="grid">
 <div data-tip="Thin prism cell layers grown on the part surface to resolve the boundary layer, which improves drag and separation. 0 = off, 3–5 is typical. Layers can fail to grow at sharp edges."><label>Boundary layers (0 = off)</label><input id="layers" type="number" value="0" min="0" max="30"></div>
 <div data-tip="How much thicker each boundary layer is than the one below it, moving away from the wall."><label>Layer expansion ratio</label><input id="expansion" type="number" value="1.2" step="0.05" min="1"></div>
 <div data-tip="Relative: layer thickness follows the local cell size (robust, the default). From y+: the first layer gets the absolute height estimated for the target y+ below, so the wall resolution matches the flow speed."><label>Layer sizing</label><select id="layer_mode"><option value="relative" selected>Relative to cell size</option><option value="yplus">First layer from y+</option></select></div>
 <div data-tip="Target dimensionless wall distance of the first cell centre. The wall function used here works across the range; 30–100 is typical for wall-function meshes, about 1 resolves the viscous sublayer (needs many thin layers). Estimated from a flat-plate correlation at the part's Reynolds number."><label>Target y+</label><input id="yplus" type="number" value="30" step="any" min="0"></div>
</div>
</div><div class="effects">
<div class="sub first">First layer</div>
<div id="yplusinfo" class="note">Load an STL to estimate the first layer height.</div>
<div class="actions"><button class="sec sm" onclick="useSuggestedLayers()" data-tip="Set Boundary layers to the count that grows from the y+ first layer to about half the surface cell size at this expansion ratio.">Use suggested layers</button></div>
</div></div>
<div class="split"><div class="inputs">
<div class="sub first">Domain and refinement</div>
<div class="note first">The flow domain around the part and the extra refinement behind it.</div>
<div class="mid-only">
<div class="grid">
 <div data-tip="Margins: the domain follows the part, with gaps given in multiples of the part size (keeps working when the part changes). Box coordinates: you give the domain's corners in metres, e.g. to match a wind-tunnel test section."><label>Domain size</label><select id="domain_mode"><option value="relative" selected>Margins around the part</option><option value="absolute">Box coordinates (m)</option></select></div>
 <div data-when="domain_mode=relative" data-tip="Gap between the part and each side wall, in part sizes. Too small and the walls squeeze the flow and raise drag (blockage)."><label>Sides (x part size)</label><input id="pad_side" type="number" value="3" step="0.5"></div>
 <div data-when="domain_mode=relative" data-tip="Gap between the inlet and the part, in part sizes."><label>Upstream (x part size)</label><input id="pad_up" type="number" value="3" step="0.5"></div>
 <div data-when="domain_mode=relative" data-tip="Gap between the part and the outlet, in part sizes. Leave room for the wake to develop before it reaches the outlet."><label>Downstream (x part size)</label><input id="pad_down" type="number" value="6" step="0.5"></div>
</div>
<div class="grid" data-when="domain_mode=absolute">
 <div data-tip="Lower X edge of the domain, in metres in the part's own coordinates (after the STL unit conversion). Must lie below the part."><label>X min (m)</label><input id="dom_x0" type="number" step="any"></div>
 <div data-tip="Upper X edge of the domain, in metres. Must lie beyond the part."><label>X max (m)</label><input id="dom_x1" type="number" step="any"></div>
 <div data-tip="Lower Y edge of the domain, in metres in the part's own coordinates. Must lie below the part."><label>Y min (m)</label><input id="dom_y0" type="number" step="any"></div>
 <div data-tip="Upper Y edge of the domain, in metres. Must lie beyond the part."><label>Y max (m)</label><input id="dom_y1" type="number" step="any"></div>
 <div data-tip="Lower Z edge of the domain, in metres in the part's own coordinates. Must lie below the part."><label>Z min (m)</label><input id="dom_z0" type="number" step="any"></div>
 <div data-tip="Upper Z edge of the domain, in metres. Must lie beyond the part."><label>Z max (m)</label><input id="dom_z1" type="number" step="any"></div>
</div>
<div class="note" data-when="domain_mode=absolute">Switching to box coordinates starts from the current domain. The inlet and outlet are still the faces across the flow direction, and the wake box is clipped to the domain. With a ground on, the floor is the min face on the up axis exactly where you put it, and may cut slightly into the part; Ground clearance is not used.</div>
<div class="sub">Wake refinement</div>
<div class="grid">
 <div data-tip="Extra refinement in a box behind the part so the wake is resolved. 0 = off. The box shows in amber in the domain preview."><label>Wake refinement level (0 = off)</label><input id="wake_level" type="number" value="0" min="0" max="6"></div>
 <div data-tip="How far the wake box reaches downstream from the back of the part, in multiples of the part size."><label>Wake length (x part size)</label><input id="wake_len" type="number" value="3" step="0.5" min="0"></div>
</div>
</div>
<div class="adv-only">
<div class="sub">Meshing controls</div>
<div class="grid">
 <div data-tip="Buffer cells kept between two refinement levels. More gives smoother size transitions, at the cost of more cells."><label>Cells between levels</label><input id="n_between" type="number" value="3" min="1" max="10"></div>
 <div data-tip="Where the surface bends more sharply than this angle, cells are refined to the full surface level so edges and curves stay crisp. Lower refines more."><label>Resolve feature angle (&deg;)</label><input id="resolve_angle" type="number" value="30" min="0" max="180"></div>
 <div data-tip="Edges where neighbouring STL faces meet at less than this angle are extracted as feature lines the mesh snaps to. 150° catches most real edges; lower ignores gentle ones."><label>Feature included angle (&deg;)</label><input id="feature_angle" type="number" value="150" min="0" max="180"></div>
 <div data-tip="Upper limit on the total number of cells while refining. snappyHexMesh stops refining once it's reached, so it protects your memory. A background mesh already bigger than this is refused."><label>Max cells</label><input id="max_cells" type="number" value="5000000" step="500000" min="10000"></div>
 <div data-tip="Thickness of the outermost boundary layer as a fraction of the neighbouring cell size. Only used with Relative layer sizing."><label>Final layer thickness (rel.)</label><input id="final_layer" type="number" value="0.3" step="0.05" min="0.01"></div>
 <div data-tip="Where layers would be thinner than this they're removed instead: a fraction of the cell size (Relative sizing) or of the first layer height (y+ sizing)."><label>Min layer thickness (rel.)</label><input id="min_thick" type="number" value="0.1" step="0.05" min="0.01"></div>
</div>
</div>
</div><div class="effects">
<div class="actions first"><div id="dommode"></div><span class="spacer"></span><div id="domviews"></div></div>
<canvas id="domview" width="720" height="380" class="stlcanvas"></canvas>
<div class="note">The flow domain (translucent) around the part. The inlet face is tinted in the flow-axis colour, the ground face is labelled when a ground is on, and the wake refinement box shows in amber when it's enabled. Probe points (P1, …) show as dots and sample lines (L1, …) as dashed lines.</div>
<div id="livepreview" class="note">Load an STL above to see the computed domain and mesh size here.</div>
</div></div>
</section>

<section class="card" id="runcard"><div class="card-h"><h2>Run</h2><p>Write the case, mesh it and solve</p></div>
<div id="modbanner" class="alert info" data-ic="info" style="display:none"><span id="modtext"></span><span class="spacer"></span><button class="sec sm" onclick="revertCase()" data-tip="Put every changed setting back to what the written case files were created with.">Revert to written case</button></div>
<div class="split mid-only"><div class="inputs">
<div class="feature panel sunken first" data-feature="probes"><div class="feature-h"></div><div class="feature-body">
<div class="note first">Record the flow at points (a history over the run) and along lines (a profile of the last result), in mm in the part's coordinates after the STL units — the frame of the domain preview.</div>
<div class="grid">
 <div data-tip="Points where U and p are recorded every iteration or time step, e.g. in the wake to watch it settle or oscillate. Three numbers in mm per point, points separated by ';' (x y z; x y z). Blank = none."><label>Probe points (mm)</label><input id="probe_points" type="text" placeholder="x y z; x y z"></div>
 <div data-tip="Straight lines U and p are sampled along whenever fields are written, e.g. across the wake behind the part. Start and end point in mm per line, lines separated by ';' (x0 y0 z0, x1 y1 z1; …). Lines can also be added after solving with Sample lines in Results. Blank = none."><label>Sample lines (mm)</label><input id="probe_lines" type="text" placeholder="x0 y0 z0, x1 y1 z1"></div>
 <div data-tip="How many evenly spaced points each sample line is evaluated at."><label>Points per line</label><input id="probe_npts" type="number" value="100" min="2" max="10000"></div>
</div>
</div></div>
</div><div class="effects">
<div data-feat="probes">
<div class="sub first">Probe locations</div>
<div id="probeinfo" class="note">No probes. They also show as markers in the domain preview.</div>
<div id="probewarn" style="display:none"></div>
</div>
</div></div>
<div class="split"><div class="inputs">
<div class="sub first">Solver</div>
<div class="grid" data-when="sim_type=external">
 <div data-tip="Maximum number of steady solver iterations. Auto-stop or residual targets can end the run earlier."><label>Iterations</label><input id="endtime" type="number" value="500"></div>
 <div class="adv-only" data-tip="Save the full flow fields every this many iterations, for ParaView. Forces and coefficients are logged every iteration regardless."><label>Write interval (iterations)</label><input id="write_interval" type="number" value="100" min="1"></div>
</div>
<div class="grid" data-when="sim_type=transient">
 <div data-tip="Simulated time to run, in seconds. Blank: three flow-throughs of the domain, usually enough for a wake to develop and shed several cycles. Results are averaged over the last part of it."><label>End time (s)</label><input id="end_time_s" type="number" step="any" min="0" placeholder="auto"></div>
 <div data-tip="How many times the full flow fields are saved over the run, for ParaView and the sample lines; evenly spaced in time. Forces are logged every time step regardless."><label>Field writes</label><input id="write_count" type="number" value="50" min="1" max="5000"></div>
 <div class="mid-only" data-tip="Largest Courant number allowed: how many cells the flow may cross per time step. The time step adapts to it. 1 is safe for this solver; up to about 5 runs faster but can smear the wake or diverge."><label>Max Courant number</label><input id="max_co" type="number" value="1" step="0.1" min="0.05"></div>
 <div class="mid-only" data-tip="Solved fields from an earlier run used as the starting flow instead of a uniform stream, so the start-up transient is shorter. Pick a solved steady case on the same domain and mesh settings (e.g. this case's template source). Fields of a different turbulence model are left uniform."><label>Initialise from case</label><select id="init_case"><option value="">Uniform free stream</option></select></div>
 <div class="adv-only" data-tip="Time step of the first step, in seconds; the solver adapts it to the Courant limit afterwards. Blank: half the Courant limit on a surface cell."><label>First time step (s)</label><input id="delta_t" type="number" step="any" min="0" placeholder="auto"></div>
 <div class="adv-only" data-tip="Time discretisation. Euler (first order) is robust; backward (second order) keeps oscillations sharper at the same time step."><label>Time scheme</label><select id="time_scheme"><option value="Euler" selected>Euler (1st order)</option><option value="backward">Backward (2nd order)</option></select></div>
 <div class="adv-only" data-tip="PIMPLE outer correctors per time step. 1 is plain PISO (needs Courant ≤ about 1); more allow larger time steps at more cost per step."><label>Outer correctors</label><input id="n_outer" type="number" value="1" min="1" max="50"></div>
 <div class="adv-only" data-tip="Pressure corrections per outer corrector. 2 is standard."><label>Pressure correctors</label><input id="n_corr" type="number" value="2" min="1" max="10"></div>
</div>
<div id="initnote" class="alert warn" style="display:none"></div>
<div class="adv-only" data-when="sim_type=external">
<div class="grid">
 <div data-tip="Fraction of each iteration's velocity update that's applied (0–1). Lower is more stable but converges slower. Blank keeps the tutorial's 0.5."><label>Relaxation U</label><input id="relax_U" type="number" step="0.05" min="0" max="1" placeholder="0.5 (tutorial)"></div>
 <div data-tip="Fraction of each iteration's pressure update that's applied (0–1). Lower is more stable but converges slower. Blank keeps the tutorial's 0.3."><label>Relaxation p</label><input id="relax_p" type="number" step="0.05" min="0" max="1" placeholder="0.3 (tutorial)"></div>
 <div data-tip="Fraction of each iteration's turbulence update that's applied (0–1), for every field of the turbulence model (ν̃, or k, ω and the transition fields). Blank keeps 0.5."><label>Relaxation turbulence</label><input id="relax_nuTilda" type="number" step="0.05" min="0" max="1" placeholder="0.5 (tutorial)"></div>
 <div data-tip="Stop the solver once the pressure residual (how much p still changes per iteration) falls below this, together with U's target. Blank = off."><label>Residual target p</label><input id="res_p" type="number" step="any" placeholder="off"></div>
 <div data-tip="Stop the solver once the velocity residual falls below this, together with p's target. Blank = off."><label>Residual target U</label><input id="res_U" type="number" step="any" placeholder="off"></div>
</div>
<div class="note">Leave a numerics field blank to keep the tutorial's value. A residual target stops the solver once both p and U fall below their targets.</div>
</div>
<div class="actions"><button data-ic="file" onclick="setup()" data-tip="Write the OpenFOAM case folder from every setting above (Physics, Mesh, Solver). Replaces any existing mesh and results in this case.">Create case files</button><span id="setupinfo" class="note"></span></div>

<div class="sub">Mesh and solve</div>
<div class="grid">
 <div data-tip="Number of MPI processes. More than 1 splits the domain into pieces that run in parallel, then reassembles the results. One per physical core is fastest; hyperthreads don't help."><label id="nprocslabel">Processes (1 = serial)</label><input id="nprocs" type="number" min="1"></div>
 <div data-tip="If checkMesh reports failed quality checks, raise the surface refinement level and mesh again from scratch."><label class="check"><input type="checkbox" id="auto_retry"> Retry on failed mesh check</label></div>
 <div data-when="sim_type=external" data-tip="Watch Cd while solving and stop cleanly, saving the fields, once it has stopped changing. Not for transient runs, which run to their end time."><label class="check"><input type="checkbox" id="autostop_enabled"> Stop when Cd converges</label></div>
</div>
<div class="grid mid-only">
 <div data-tip="How many times to re-mesh after a failed mesh check before giving up."><label>Max mesh retries</label><input id="max_retries" type="number" value="2" min="0" max="10"></div>
 <div data-tip="How much the surface refinement level goes up on each retry."><label>Level bump per retry</label><input id="level_bump" type="number" value="1" min="1" max="3"></div>
 <div data-when="sim_type=external" data-tip="Cd counts as converged when (max − min) ÷ mean over the recent window is below this. 0.01 = within 1 %."><label>Auto-stop tolerance (rel. Cd spread)</label><input id="as_tol" type="number" value="0.01" step="0.001"></div>
 <div data-when="sim_type=external" data-tip="Never auto-stop before this iteration, so the start-up transient can't look converged."><label>Min iterations</label><input id="as_miniter" type="number" value="100" min="0"></div>
</div>
<div class="grid adv-only" data-when="sim_type=external">
 <div data-tip="Number of most recent Cd values the spread is measured over."><label>Auto-stop window (samples)</label><input id="as_window" type="number" value="20" min="5"></div>
 <div data-tip="How many checks in a row must pass before stopping, so a brief flat spot doesn't end the run."><label>Consecutive checks</label><input id="as_consec" type="number" value="3" min="1"></div>
 <div data-tip="Seconds between convergence checks."><label>Poll interval (s)</label><input id="as_poll" type="number" value="5" min="1"></div>
</div>
<div class="actions">
 <button class="sec" data-ic="mesh" onclick="act('mesh')" data-tip="Build the mesh: background block mesh, feature edges, snappyHexMesh around the part, then a checkMesh quality check. Waits in the queue if another job is running.">Mesh</button>
 <button class="sec" data-ic="play" onclick="act('solve')" data-tip="Run the solver on the existing mesh. Previous results in this case are cleared first.">Solve</button>
 <button data-ic="both" onclick="act('all')" data-tip="Mesh, then solve, but only if the mesh passes its quality check.">Mesh + Solve</button>
 <button class="danger" data-ic="stop" onclick="act('stop')" data-tip="During solving, the first click asks the solver to save its current fields and stop; a second click kills it. Other jobs stop immediately. Queued jobs still run.">Stop</button>
 <span class="spacer"></span>
 <button class="ghost sm" onclick="$('nprocs').value=CORES[0]" data-tip="Set Processes to the number of physical CPU cores.">Max processes</button>
 <button class="ghost sm" id="notifybtn" onclick="enableNotify()" data-tip="Show a browser notification and play a beep when a job finishes.">Enable notifications</button>
</div>
</div><div class="effects">
<div class="sub first">Progress</div>
<div id="timeinfo" data-when="sim_type=transient"></div>
<div id="progwrap" style="display:none"><progress id="solveprog" max="1" value="0"></progress><div id="progtext" class="note"></div></div>
<div id="queue" class="note"></div>
<div id="meshstatus" class="note">No mesh run yet.</div>
<div id="meshwarn" style="display:none"></div>
<div id="meshstats"></div>
<details open><summary>Solver log</summary><pre id="log"></pre></details>
</div></div>
</section>

<section class="card locked" id="rescard"><div class="card-h"><h2>Results</h2><p>ParaView views, forces and coefficients</p><span class="spacer"></span><button class="sec sm" onclick="refreshCaseView()" data-tip="Reload this case's results, charts and mesh status. They also refresh on their own while a job runs.">Refresh</button></div>
<div class="lock-note note">Run the solver to see results here.</div>
<div class="split"><div class="inputs">
<div class="actions first"><button class="sec sm" onclick="exportCase('export.csv')" data-tip="Download every iteration's coefficients, forces (N) and moments (N m) as a CSV file for a spreadsheet.">Export CSV</button><button class="sec sm" onclick="exportCase('report')" data-tip="Open a printable report of this case in a new tab: settings, domain, mesh quality, averaged results and convergence charts. Print it to save a PDF.">Case report</button></div>
<div class="sub">Open in ParaView</div>
<div class="grid">
 <div data-tip="Flow pathlines only: shows just the pathlines that pass within this distance of the part's surface and hides the rest, so the flow that actually meets the part stands out. Blank shows them all."><label>Pathlines near the part (mm)</label><input id="pv_near_mm" type="number" min="0" placeholder="all"></div>
</div>
<div id="pvbtns" class="tiles"></div>
<div class="note">Pressure is shown as Cp = p / (&frac12; U&sup2;) and velocity as |U|, each coloured over its own range at the shown iteration.</div>
<div class="sub">Forces and coefficients</div>
<div class="grid">
 <div data-when="sim_type=external" data-tip="A steady run still wobbles a little at the end, so results are reported as the mean over this many final iterations, with their min–max range. Larger smooths more; it shouldn't reach back into the start-up transient."><label>Average over last (iterations)</label><input id="avg_window" type="number" value="100" min="1"></div>
 <div data-when="sim_type=transient" data-tip="Transient results are time-weighted means over this many final seconds of simulated time, with their min–max range, and the Strouhal number is measured in the same window. Blank: the last half of the run. Keep it clear of the start-up transient and long enough for several shedding cycles."><label>Average over last (s)</label><input id="avg_time" type="number" step="any" min="0" placeholder="last half"></div>
</div>
<div class="mid-only">
<div data-feat="probes">
<div class="sub">Probes</div>
<div class="note first">Point probes record during the solve. Sample lines are evaluated on the latest result, so lines added or changed under Run → Probes after solving can be sampled without solving again.</div>
<div class="actions"><button class="sec sm" onclick="act('sample')" data-tip="Sample U and p along the sample lines set under Run → Probes, on this case's latest result. Queued like a job; takes seconds.">Sample lines</button></div>
</div>
<div class="feature panel sunken" data-feature="compare"><div class="feature-h"></div><div class="feature-body"><div class="grid">
 <div data-tip="Another solved case to compare with: its averaged coefficients and forces side by side with this case's, the settings that differ, and its coefficient history overlaid on the convergence chart. Cases in the same folder are listed first."><label>Compare with</label><select id="cmp_case"><option value="">None</option></select></div>
</div></div></div>
</div>
</div><div class="effects">
<div id="avgstats"></div>
<div id="forceline" class="note"></div>
<pre id="results" class="results">Results appear here.</pre>
<details><summary>Coefficient history (last 50 iterations)</summary><div id="coefftable" class="scroll"></div></details>
<div class="sub row">Convergence<span class="spacer"></span><div id="coefseg" class="seg"></div></div>
<canvas id="chart" width="600" height="210" class="chartcanvas" data-tip="The chosen coefficient per iteration. A flat line means it has converged. The vertical range ignores the first 10 % of iterations so the start-up spike doesn't flatten the rest."></canvas>
<canvas id="reschart" width="600" height="230" class="chartcanvas" data-tip="Initial residual of each field per iteration, on a log scale: how much the solution still changes. Falling 3–4 decades and levelling off is typical of a converged run."></canvas>
<div class="mid-only">
<div data-feat="probes">
<div class="sub row">Probes<span class="spacer"></span><div id="probeqseg" class="seg"></div></div>
<div id="probenote" class="note">No probe data. Set probe points or sample lines under Run → Probes.</div>
<canvas id="linechart" width="600" height="210" class="chartcanvas" style="display:none" data-tip="The chosen quantity along each sample line at the latest result, against the distance from the line's start point."></canvas>
<canvas id="probechart" width="600" height="210" class="chartcanvas" style="display:none" data-tip="The chosen quantity at each probe point over the run: settling to a constant means a steady flow there, a regular oscillation means shedding."></canvas>
</div>
<div data-feat="compare">
<div class="sub">Compare</div>
<div id="cmpnote" class="note">Pick a case under Compare with to see it next to this one.</div>
<div id="cmptable" class="scroll"></div>
<details id="cmpdiffbox" style="display:none"><summary>Settings that differ</summary><div id="cmpdiff" class="scroll"></div></details>
</div>
</div>
</div></div>
</section>

<section class="card mid-only" id="studycard"><div class="card-h"><h2>Studies</h2><p>Parameter sweeps and mesh independence, one new case per value</p><span class="spacer"></span><div class="seg" data-fseg="studies"></div></div>
<div class="split" data-feat="studies"><div class="inputs">
<div class="grid">
 <div data-tip="The setting to vary across cases. Mesh resolution, cell size or cell count gives a mesh-independence study: Cd should settle as the cell count grows. Everything else is copied from the form."><label>Vary</label><select id="sweep_param"><option value="speed">Flow speed (m/s)</option><option value="dir">Flow direction</option><option value="aoa">Angle of attack (&deg;)</option><option value="yaw">Yaw (&deg;)</option><option value="mesh_preset">Mesh resolution (coarse, medium, fine, very fine)</option><option value="cell_size_mm">Background cell size (mm)</option><option value="surface_cell_mm">Surface cell size (mm)</option><option value="target_cells">Background cell count</option></select></div>
 <div data-tip="Comma-separated values. Each becomes its own case (name_sweep1, name_sweep2, …), meshed and solved in turn."><label>Values (comma-separated)</label><input id="sweep_values" placeholder="e.g. 10,15,20"></div>
 <div data-feat="aircraft" data-tip="The static margin you want, in percent of the mean chord: how far the CG sits ahead of the neutral point. 5–15 % is typical for RC models; more is steadier but needs more elevator. Used to suggest a CG position after an angle-of-attack sweep."><label>Target static margin (%)</label><input id="target_sm" type="number" value="10" step="1"></div>
</div>
<div class="note">Angle of attack sweeps give the polars, the neutral point and, with an aircraft mass, flight speeds. Yaw sweeps (full model) give the directional and roll stability. Angle and speed sweeps mesh once and reuse that mesh.</div>
<div class="actions"><button onclick="runSweep()" data-tip="Queue the sweep. Each variant's results are averaged over the last iterations set under Results.">Run sweep</button></div>
</div><div class="effects">
<div id="sweepprogress" class="note">No sweep run yet.</div>
<div id="sweeptable" class="scroll"></div>
<div id="aerostats" data-feat="aircraft"></div>
<div id="aeronote" class="note" data-feat="aircraft"></div>
<div class="actions"><div id="sweepseg" class="seg"></div></div>
<canvas id="sweepchart" width="600" height="210" class="chartcanvas" style="display:none" data-tip="The sweep's results per variant. For a mesh study Cd is plotted against the cell count: once it stops changing as cells are added, the coarser mesh is fine enough. Angle sweeps offer the polars, yaw sweeps the stability curves."></canvas>
</div></div>
</section>
</div>
</main>
<script>
const $=id=>document.getElementById(id);let pos=0,tick=0;
/* per-case feature switches (FEATURE_GROUPS server-side: key -> [label, {param: neutral value}]). #features holds
   the switched-off keys. A switched-off feature's fields keep their values (saved with the case) but aren't used:
   fv(id) is a field's value as used - the neutral value while its feature is off - mirroring apply_features() */
const FEATURE_GROUPS=/*FEATURE_GROUPS*/null;
const FEAT_OF={};Object.entries(FEATURE_GROUPS).forEach(([k,[,ps]])=>Object.keys(ps).forEach(id=>FEAT_OF[id]=k));
const featOff=()=>$('features').value.split(' ').filter(k=>k in FEATURE_GROUPS);
const featOn=k=>!featOff().includes(k);
const fv=id=>{const k=FEAT_OF[id];return k&&!featOn(k)?FEATURE_GROUPS[k][1][id]:$(id).value;};
/* a params object as used: switched-off features' params neutral (for saved params of other cases too) */
const effParams=p=>{const o={...p},off=String(p.features||'').split(' ');
  off.forEach(k=>{if(FEATURE_GROUPS[k])Object.assign(o,FEATURE_GROUPS[k][1]);});return o;};
async function j(url,opt){const r=await fetch(url,opt);return r.json();}
/* ---------- icons: inline SVG paths on a 24-unit grid (derived from Lucide, ISC licence) ---------- */
const ICONS={check:'<path d="M20 6 9 17l-5-5"/>',x:'<path d="M18 6 6 18M6 6l12 12"/>',
  alert:'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/>',
  info:'<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  upload:'<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5M12 3v12"/>',
  play:'<path d="M6 3l14 9-14 9z"/>',stop:'<rect width="14" height="14" x="5" y="5" rx="2"/>',
  mesh:'<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
  both:'<path d="M13 19l9-7-9-7zM2 19l9-7-9-7z"/>',
  file:'<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/><path d="M14 2v4a2 2 0 0 0 2 2h4M9 15h6M12 18v-6"/>',
  external:'<path d="M15 3h6v6M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
  folder:'<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>'};
const icon=(name,cls)=>'<svg class="ic'+(cls?' '+cls:'')+'" viewBox="0 0 24 24" aria-hidden="true">'+ICONS[name]+'</svg>';
document.querySelectorAll('[data-ic]').forEach(el=>el.insertAdjacentHTML('afterbegin',icon(el.dataset.ic)));
/* .alert into el: kind warn|bad|info, html message (already escaped); hidden when there's no message */
const ALERT_IC={warn:'alert',bad:'x',info:'info'};
function setAlert(el,kind,html){
  el.style.display=html?'':'none';
  if(html){el.className='alert '+kind;el.innerHTML=icon(ALERT_IC[kind])+'<span>'+html+'</span>';}
}
/* a .badge, with a check / x icon for ok / bad */
const badge=(text,kind)=>'<span class="badge'+(kind?' '+kind:'')+'">'+(kind?icon(kind==='ok'?'check':'x'):'')+esc(text)+'</span>';

/* ---------- STL viewers (hand-rolled canvas, no CDN): geometry + domain preview ---------- */
/* 3x3 row-major rotation helpers. View matrix rows = screen right / up / toward-viewer, in model axes. */
function matMul(a,b){const r=[];for(let i=0;i<3;i++)for(let k=0;k<3;k++)r.push(a[i*3]*b[k]+a[i*3+1]*b[3+k]+a[i*3+2]*b[6+k]);return r;}
function rx(t){const c=Math.cos(t),s=Math.sin(t);return [1,0,0, 0,c,-s, 0,s,c];}
function ry(t){const c=Math.cos(t),s=Math.sin(t);return [c,0,s, 0,1,0, -s,0,c];}
const VIEWS={iso:matMul(rx(-0.5),ry(0.6)),
  '+X':[0,1,0, 0,0,1, 1,0,0], '-X':[0,-1,0, 0,0,1, -1,0,0],
  '+Y':[-1,0,0, 0,0,1, 0,1,0], '-Y':[1,0,0, 0,0,1, 0,-1,0],
  '+Z':[1,0,0, 0,1,0, 0,0,1], '-Z':[-1,0,0, 0,1,0, 0,0,-1]};
/* one colour per model axis, shared by the axis triad, the flow arrow and the page legend */
/* data colours are design tokens (:root --ax-*, --geo, --wake, --c1..--c6), read once; canvases need #rrggbb */
const tok=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const AXIS_COL={x:tok('--ax-x'),y:tok('--ax-y'),z:tok('--ax-z')},GEO_COL=tok('--geo'),WAKE_COL=tok('--wake');
function makeViewer(canvas,opts){
  opts=opts||{};
  const ctx=canvas.getContext('2d'),aspect=canvas.height/canvas.width,baseW=canvas.width;
  let verts=[],boxes=[],marks=[],center=[0,0,0],radius=1,R=VIEWS.iso.slice(),mode='solid',zoom=opts.zoom||1,dragging=false,lastX=0,lastY=0;
  const apply=v=>[R[0]*v[0]+R[1]*v[1]+R[2]*v[2],R[3]*v[0]+R[4]*v[1]+R[5]*v[2],R[6]*v[0]+R[7]*v[1]+R[8]*v[2]];
  function rot(p){return apply([p[0]-center[0],p[1]-center[1],p[2]-center[2]]);}
  const css=n=>getComputedStyle(document.body).getPropertyValue(n).trim()||'#888';
  function drawTriad(w,h){
    const ox=30,oy=h-30,axes=[['X',AXIS_COL.x,[1,0,0]],['Y',AXIS_COL.y,[0,1,0]],['Z',AXIS_COL.z,[0,0,1]]]
      .map(a=>[a[0],a[1],apply(a[2])]).sort((p,q)=>p[2][2]-q[2][2]);
    ctx.lineWidth=2;ctx.lineCap='round';ctx.font='bold 11px system-ui';
    for(const [n,col,v] of axes){
      ctx.strokeStyle=ctx.fillStyle=col;
      ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(ox+v[0]*22,oy-v[1]*22);ctx.stroke();
      ctx.fillText(n,ox+v[0]*28-4,oy-v[1]*28+4);
    }
    ctx.lineWidth=1;
  }
  function drawFlow(proj){
    const f=opts.flowDir&&opts.flowDir();if(!f)return;   /* {d: unit vector, ax: 'x'|'y'|'z', label} */
    const d=f.d;
    const at=k=>proj(rot([center[0]-d[0]*radius*k,center[1]-d[1]*radius*k,center[2]-d[2]*radius*k]));
    const a=at(1.4),b=at(1.04),dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy);
    const col=AXIS_COL[f.ax];ctx.strokeStyle=ctx.fillStyle=col;ctx.lineWidth=3;ctx.lineCap='round';
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.lineWidth=1;
    if(L>1){  /* arrowhead; skipped when the arrow points straight at/away from the viewer */
      const ux=dx/L,uy=dy/L;
      ctx.beginPath();ctx.moveTo(b[0]+ux*8,b[1]+uy*8);ctx.lineTo(b[0]-uy*6,b[1]+ux*6);ctx.lineTo(b[0]+uy*6,b[1]-ux*6);ctx.closePath();ctx.fill();
    }
    ctx.font='11px system-ui';ctx.fillText('flow '+f.label,a[0]+4,a[1]-6);
  }
  const BASE=[148,163,184];   /* neutral blue-grey material */
  let img=null,depth=null;    /* z-buffer render target, reallocated only on resize */
  /* Rasterize one triangle (pixel-space x,y + depth z; larger z = closer) into img/depth.
     opaque: depth-test and write depth. translucent: depth-test only, "over"-composite. */
  function raster(a,b,c,col,alpha,opaque){
    const W=img.width,H=img.height,px=img.data;
    const area=(b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1]);
    if(Math.abs(area)<1e-9)return;
    const x0=Math.max(0,Math.floor(Math.min(a[0],b[0],c[0]))),x1=Math.min(W-1,Math.ceil(Math.max(a[0],b[0],c[0])));
    const y0=Math.max(0,Math.floor(Math.min(a[1],b[1],c[1]))),y1=Math.min(H-1,Math.ceil(Math.max(a[1],b[1],c[1])));
    const d0=(b[1]-c[1])/area,d1=(c[1]-a[1])/area,E=-1e-7;   /* d(weight)/dx, constant per triangle */
    for(let y=y0;y<=y1;y++){
      const py=y+0.5,sx=x0+0.5;
      let w0=((b[0]-sx)*(c[1]-py)-(c[0]-sx)*(b[1]-py))/area,w1=((c[0]-sx)*(a[1]-py)-(a[0]-sx)*(c[1]-py))/area;
      for(let x=x0;x<=x1;x++,w0+=d0,w1+=d1){
        const w2=1-w0-w1;
        if(w0<E||w1<E||w2<E)continue;
        const z=w0*a[2]+w1*b[2]+w2*c[2],i=y*W+x;
        if(z<=depth[i])continue;
        const k=i*4;
        if(opaque){depth[i]=z;px[k]=col[0];px[k+1]=col[1];px[k+2]=col[2];px[k+3]=255;continue;}
        const da=px[k+3]/255,oa=alpha+da*(1-alpha),f=da*(1-alpha);
        px[k]=(col[0]*alpha+px[k]*f)/oa;px[k+1]=(col[1]*alpha+px[k+1]*f)/oa;px[k+2]=(col[2]*alpha+px[k+2]*f)/oa;px[k+3]=oa*255;
      }
    }
  }
  /* box helpers: corner i picks min/max per axis from its bits; faces keyed '-x','+x',... */
  const corner=(b,i)=>[i&1?b.max[0]:b.min[0],i&2?b.max[1]:b.min[1],i&4?b.max[2]:b.min[2]];
  const FACES={'-x':[0,2,6,4],'+x':[1,3,7,5],'-y':[0,1,5,4],'+y':[2,3,7,6],'-z':[0,1,3,2],'+z':[4,5,7,6]};
  const hex=s=>[1,3,5].map(i=>parseInt(s.slice(i,i+2),16));
  function render(){
    /* backing store at 1x CSS pixels: the software rasterizer's cost grows with ss², so speed beats edge smoothing */
    const ss=1,w=canvas.clientWidth||baseW,h=Math.round(w*aspect);
    const W=Math.round(w*ss),H=Math.round(h*ss);
    if(canvas.width!==W||canvas.height!==H){canvas.width=W;canvas.height=H;img=null;}
    ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,W,H);
    if(!verts.length&&!boxes.length)return;
    const scale=(Math.min(w,h)*0.33*zoom)/radius,S=scale*ss;
    const proj=p=>[w/2+p[0]*scale,h/2-p[1]*scale];
    const toPix=q=>[W/2+q[0]*S,H/2-q[1]*S,q[2]],pix=p=>toPix(rot(p));
    const wire=mode==='wire',fg=css('--fg');
    const ll=Math.hypot(0.35,0.45,0.82),L=[0.35/ll,0.45/ll,0.82/ll];
    const hl=Math.hypot(L[0],L[1],L[2]+1),Hv=[L[0]/hl,L[1]/hl,(L[2]+1)/hl];   /* Blinn half-vector, viewer at +z */
    if(!wire){
      if(!img){img=ctx.createImageData(W,H);depth=new Float32Array(W*H);}
      img.data.fill(0);depth.fill(-Infinity);
      for(let i=0;i+2<verts.length;i+=3){
        const ra=rot(verts[i]),rb=rot(verts[i+1]),rc=rot(verts[i+2]);
        /* two-sided normal: STL winding is often inconsistent */
        const ux=rb[0]-ra[0],uy=rb[1]-ra[1],uz=rb[2]-ra[2],vx=rc[0]-ra[0],vy=rc[1]-ra[1],vz=rc[2]-ra[2];
        let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
        const nl=Math.hypot(nx,ny,nz)||1,sg=nz<0?-1:1;nx*=sg/nl;ny*=sg/nl;nz*=sg/nl;
        const k=0.3+0.7*Math.max(0,nx*L[0]+ny*L[1]+nz*L[2]),s=64*Math.pow(Math.max(0,nx*Hv[0]+ny*Hv[1]+nz*Hv[2]),40);
        raster(toPix(ra),toPix(rb),toPix(rc),[Math.min(255,BASE[0]*k+s),Math.min(255,BASE[1]*k+s),Math.min(255,BASE[2]*k+s)],1,true);
      }
      for(const bx of boxes){   /* translucent faces after the opaque part, so the part shows through */
        for(const [key,q] of Object.entries(FACES)){
          const f=(bx.faces&&bx.faces[key])||bx,col=hex(f.color),p=q.map(i=>pix(corner(bx,i)));
          raster(p[0],p[1],p[2],col,f.alpha,false);raster(p[0],p[2],p[3],col,f.alpha,false);
        }
      }
      ctx.putImageData(img,0,0);
    }
    ctx.setTransform(ss,0,0,ss,0,0);
    ctx.lineJoin='round';
    if(wire){
      ctx.globalAlpha=0.5;ctx.lineWidth=0.6;ctx.strokeStyle=fg;
      for(let i=0;i+2<verts.length;i+=3){
        const pa=proj(rot(verts[i])),pb=proj(rot(verts[i+1])),pc=proj(rot(verts[i+2]));
        ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.lineTo(pc[0],pc[1]);ctx.closePath();ctx.stroke();
      }
      ctx.globalAlpha=1;
    }
    ctx.font='11px system-ui';
    const halo=(t,x,y)=>{   /* a label in --fg with a --bg halo, readable over the part */
      ctx.font='600 11px '+css('--font');ctx.lineWidth=3;ctx.lineJoin='round';ctx.strokeStyle=css('--bg');
      ctx.strokeText(t,x,y);ctx.fillStyle=fg;ctx.fillText(t,x,y);ctx.lineWidth=1;ctx.font='11px system-ui';
    };
    const pdot=s=>{ctx.fillStyle=fg;ctx.strokeStyle=css('--bg');ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(s[0],s[1],3.5,0,2*Math.PI);ctx.fill();ctx.stroke();};
    for(const m of marks){
      if(m.b){   /* sample line: dashed segment between two dots, labelled at its end */
        const a=proj(rot(m.c)),b=proj(rot(m.b));
        ctx.strokeStyle=fg;ctx.lineWidth=1.25;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.setLineDash([]);
        pdot(a);pdot(b);halo(m.label,b[0]+7,b[1]+4);continue;
      }
      if(m.kind==='probe'){pdot(proj(rot(m.c)));const s=proj(rot(m.c));halo(m.label,s[0]+7,s[1]+4);continue;}
      /* CG roundel: quartered circle at a constant screen size, drawn over everything so the part never hides it */
      const s=proj(rot(m.c)),r=7,light=fg,dark=css('--bg');
      for(let q=0;q<4;q++){
        ctx.fillStyle=q%2?dark:light;ctx.beginPath();ctx.moveTo(s[0],s[1]);
        ctx.arc(s[0],s[1],r,q*Math.PI/2-Math.PI/2,(q+1)*Math.PI/2-Math.PI/2);ctx.closePath();ctx.fill();
      }
      ctx.strokeStyle=dark;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(s[0],s[1],r+0.75,0,2*Math.PI);ctx.stroke();
      ctx.strokeStyle=light;ctx.lineWidth=1;ctx.beginPath();ctx.arc(s[0],s[1],r,0,2*Math.PI);ctx.stroke();
      ctx.font='600 11px '+css('--font');ctx.lineWidth=3;ctx.lineJoin='round';ctx.strokeStyle=dark;
      ctx.strokeText(m.label,s[0]+r+5,s[1]+4);ctx.fillStyle=light;ctx.fillText(m.label,s[0]+r+5,s[1]+4);
      ctx.lineWidth=1;ctx.font='11px system-ui';
    }
    for(const bx of boxes){   /* box edges + labels as vector overlay */
      ctx.strokeStyle=bx.color;ctx.globalAlpha=0.7;ctx.lineWidth=1;ctx.beginPath();
      for(let i=0;i<8;i++)for(const bit of [1,2,4])if(!(i&bit)){
        const p=proj(rot(corner(bx,i))),q=proj(rot(corner(bx,i|bit)));ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);
      }
      ctx.stroke();ctx.globalAlpha=1;
      const label=(t,p,col)=>{const s=proj(rot(p));ctx.fillStyle=col;ctx.fillText(t,s[0]+3,s[1]-3);};
      if(bx.label)label(bx.label,bx.max,bx.color);
      for(const [key,f] of Object.entries(bx.faces||{})){
        if(!f.label)continue;
        const q=FACES[key].map(i=>corner(bx,i));
        label(f.label,[0,1,2].map(d=>(q[0][d]+q[1][d]+q[2][d]+q[3][d])/4),f.color);
      }
    }
    drawFlow(proj);
    if(!opts.static)drawTriad(w,h);
  }
  let queued=false;
  const redraw=()=>{if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;render();});};
  /* framing: part alone -> centroid/max distance; with boxes -> union bounding box */
  function frame(){
    if(boxes.length){
      const mn=[Infinity,Infinity,Infinity],mx=[-Infinity,-Infinity,-Infinity];
      const grow=p=>{for(let d=0;d<3;d++){mn[d]=Math.min(mn[d],p[d]);mx[d]=Math.max(mx[d],p[d]);}};
      boxes.forEach(b=>{grow(b.min);grow(b.max);});verts.forEach(grow);
      center=[0,1,2].map(d=>(mn[d]+mx[d])/2);radius=Math.hypot(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2])/2||1;
    }else if(verts.length){
      let cx=0,cy=0,cz=0;verts.forEach(p=>{cx+=p[0];cy+=p[1];cz+=p[2];});
      const n=verts.length;cx/=n;cy/=n;cz/=n;
      let r=0;verts.forEach(p=>{r=Math.max(r,Math.hypot(p[0]-cx,p[1]-cy,p[2]-cz));});
      center=[cx,cy,cz];radius=r||1;
    }
  }
  if(!opts.static){   /* opts.static: a thumbnail, drawn once at its size, no interaction */
    window.addEventListener('resize',redraw);
    canvas.onmousedown=e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;};
    window.addEventListener('mouseup',()=>dragging=false);
    window.addEventListener('mousemove',e=>{
      if(!dragging)return;
      /* pre-multiply: both drag directions turn about fixed screen axes, not the model's own */
      R=matMul(matMul(ry((e.clientX-lastX)*0.01),rx((e.clientY-lastY)*0.01)),R);
      lastX=e.clientX;lastY=e.clientY;redraw();
    });
    canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(0.2,Math.min(5,zoom*(e.deltaY<0?1.1:0.9)));redraw();},{passive:false});
  }
  return {
    setMode(m){mode=m;redraw();},
    view(name){R=VIEWS[name].slice();redraw();},
    redraw,
    set(v){verts=v||[];frame();redraw();},
    setBoxes(b){boxes=b||[];frame();redraw();},
    /* [{c, label}]: CG roundels; {c, label, kind:'probe'}: probe dots; {c, b, label}: sample lines.
       Not part of the framing */
    setMarkers(m){marks=m||[];redraw();},
  };
}
/* Iso/+-X/+-Y/+-Z view buttons and a Solid/Wireframe toggle, built for any viewer */
function viewBar(el,viewer){
  el.className='seg';el.title='View the part from this side';
  el.innerHTML=['iso','+X','-X','+Y','-Y','+Z','-Z'].map(v=>'<button data-v="'+v+'">'+(v==='iso'?'Iso':v.replace('-','&minus;'))+'</button>').join('');
  el.querySelectorAll('button').forEach(b=>b.onclick=()=>viewer.view(b.dataset.v));
}
function modeBar(el,viewer){
  el.className='seg';
  el.innerHTML='<button class="on" data-m="solid">Solid</button><button data-m="wire">Wireframe</button>';
  el.querySelectorAll('button').forEach(b=>b.onclick=()=>{viewer.setMode(b.dataset.m);el.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));});
}
/* inflow for the viewer arrow: the dir axis tilted by angle of attack (toward the up/lift axis, liftAxis())
   and yaw (toward the remaining axis) - mirrors compute_geometry() */
/* mirrors compute_geometry(): the up_axis setting, automatic = z (y when the flow runs along z) */
const liftAxis=ax=>{const u=$('up_axis').value;return u!=='auto'?u:ax==='z'?'y':'z';};
const spanAxis=ax=>'xyz'.replace(ax,'').replace(liftAxis(ax),'');   /* the pitch axis: neither flow nor lift */
function flowDir(){
  const dir=$('dir').value,ax=dir[1],i='xyz'.indexOf(ax),li='xyz'.indexOf(liftAxis(ax)),pi=3-i-li;
  const a=(+fv('aoa')||0)*Math.PI/180,b=(+fv('yaw')||0)*Math.PI/180,d=[0,0,0];
  d[i]=Math.cos(a)*Math.cos(b)*(dir[0]==='-'?-1:1);d[li]=Math.sin(a);d[pi]=Math.cos(a)*Math.sin(b);
  const tilt=[a&&('AoA '+fv('aoa')+'°'),b&&('yaw '+fv('yaw')+'°')].filter(Boolean).join(', ');
  return {d,ax,label:dir+(tilt?' ('+tilt+')':'')};
}
const physView=makeViewer($('physview'),{flowDir});
const domView=makeViewer($('domview'),{flowDir});
const VIEWERS=[physView,domView];
[[physView,'phys'],[domView,'dom']].forEach(([v,k])=>{viewBar($(k+'views'),v);modeBar($(k+'mode'),v);});
let PART_BOX=null;   /* the previewed part's bounding box (STL file units), framing the Physics view's planes */
/* the loaded part's preview (file units) into both viewers; always the whole part, also for a half model */
function setPreview(v){
  VIEWERS.forEach(x=>x.set(v));
  PART_BOX=null;
  if(v&&v.length){PART_BOX=[0,1,2].map(()=>[Infinity,-Infinity]);
    for(const p of v)for(let d=0;d<3;d++){PART_BOX[d][0]=Math.min(PART_BOX[d][0],p[d]);PART_BOX[d][1]=Math.max(PART_BOX[d][1],p[d]);}}
}
['dir','up_axis','aoa','yaw'].forEach(id=>$(id).addEventListener('change',()=>VIEWERS.forEach(x=>x.redraw())));
/* a tilted inflow crosses the side faces: it needs far-field (freestream) sides, as setup_case() enforces */
['aoa','yaw'].forEach(id=>$(id).addEventListener('change',()=>{if(+$(id).value)$('side_bc').value='freestream';}));
/* the domain's ground and half-model symmetry faces, keyed like the viewer's box faces ('-z', '+y', ...) */
function boundaryFaces(){
  const ax=$('dir').value[1],faces={};
  if(fv('ground')!=='none')faces['-'+liftAxis(ax)]={color:GEO_COL,alpha:0.3,label:'ground'};
  const half=fv('half_model');
  if(half!=='none')faces[(half==='max'?'+':'-')+spanAxis(ax)]={color:GEO_COL,alpha:0.3,label:'symmetry'};
  return faces;
}
/* domain preview boxes from a /api/preview or /api/setup response (metres -> STL file units); the Physics
   view gets only the ground/symmetry faces of the domain and a small box at the moment reference (CG) */
function setDomainBoxes(r){
  const s=+$('scale').value||1,u=a=>a.map(v=>v/s),dir=$('dir').value,ax=dir[1],col=AXIS_COL[ax];
  const inlet=(dir[0]==='+'?'-':'+')+ax,outlet=dir[0]+ax;
  const faces={[inlet]:{color:col,alpha:0.22,label:'inlet'},[outlet]:{color:GEO_COL,alpha:0.14,label:'outlet'},...boundaryFaces()};
  const boxes=[{min:u(r.domain_min),max:u(r.domain_max),color:GEO_COL,alpha:0.08,faces}];
  if(+$('wake_level').value>0&&r.wake_min)boxes.push({min:u(r.wake_min),max:u(r.wake_max),color:WAKE_COL,alpha:0.12,label:'wake'});
  domView.setBoxes(boxes);
  const pr=r.probes||{points:[],lines:[]};   /* probe points and sample lines (metres) as markers */
  domView.setMarkers(pr.points.map(q=>({c:u(q.at),label:q.name,kind:'probe'}))
    .concat(pr.lines.map(l=>({c:u(l.start),b:u(l.end),label:l.name.replace('line','L')}))));
  /* each plane: the domain face flattened onto its axis, trimmed to half a part size around the part */
  const lo=u(r.domain_min),hi=u(r.domain_max),pb=[];
  for(const [key,f] of Object.entries(boundaryFaces())){
    const a='xyz'.indexOf(key[1]),at=key[0]==='-'?lo[a]:hi[a],mn=lo.slice(),mx=hi.slice();
    if(PART_BOX){
      const m=Math.max(...PART_BOX.map(([a0,a1])=>a1-a0))/2;
      for(let k=0;k<3;k++){mn[k]=Math.max(mn[k],PART_BOX[k][0]-m);mx[k]=Math.min(mx[k],PART_BOX[k][1]+m);}
    }
    mn[a]=mx[a]=at;
    pb.push({min:mn,max:mx,color:f.color,alpha:0.15,label:f.label});
  }
  physView.setBoxes(pb);
  physView.setMarkers([{c:u(r.CofR),label:'CG'}]);
}

/* the drop zone's status line: the chosen file (with a solid border) or an upload error */
function showFile(text,chosen){$('bbox').innerHTML=(chosen?icon('check','sm'):'')+esc(text);$('dropzone').classList.toggle('has-file',chosen);}
async function uploadFile(f){
  if(!f)return;
  const r=await j('/api/upload?name='+encodeURIComponent(f.name),{method:'POST',body:await f.arrayBuffer()});
  if(!r.ok){showFile(r.error,false);return;}
  applyStl(r,f.name,Math.max(...r.ext)>5?'0.001':'1');
}
/* an STL the server now holds as the current geometry (an upload or a folder's shared part): show it, set its units */
function applyStl(r,name,scale){
  showFile(name+' · '+big(r.tris)+' triangles · size '+r.ext.map(v=>v.toPrecision(4)).join(' x ')+' (file units)',true);
  STLNAME=name+' · '+big(r.tris)+' triangles';
  $('scale').value=scale;
  setPreview(r.preview);
  updatePreview();
}

/* live flow/mesh preview - recomputed from the STL + current settings, no case created */
/* geometry/mesh-sizing readout as stat chips (shared by /api/preview and /api/setup responses) */
let LAST_PREVIEW=null;   /* last good preview/setup response: seeds the Custom mesh fields */
let STLNAME='';          /* the loaded geometry, for the Geometry step's status line */
/* stat chips into el from [[label, value html]] (null values skipped), or a plain note when there are none */
function chips(el,rows,empty){
  rows=rows.filter(r=>r[1]!=null);
  el.classList.toggle('stats',rows.length>0);el.classList.toggle('note',!rows.length);
  if(rows.length)el.innerHTML=rows.map(([k,v])=>'<div class="stat"><b>'+k+'</b>'+v+'</div>').join('');
  else el.textContent=empty||'';
}
const mm=v=>+(v*1000).toPrecision(3)+' mm';
function showStats(r){
  if(!r.ok){
    ['livepreview','cellinfo','yplusinfo','physstats','probeinfo','timeinfo'].forEach(id=>chips($(id),[],r.error));
    /* no stale domain: the last good one no longer matches these settings */
    domView.setBoxes([]);domView.setMarkers([]);physView.setBoxes([]);physView.setMarkers([]);
    $('cellnote').textContent='';renderPhysics(null);renderSteps();return;
  }
  LAST_PREVIEW=r;
  const pm=PRESET_MM[$('mesh_preset').value];   /* a preset: show the sizes it picked */
  if(pm){$('cell_size_mm').value=+(r.cell_size_m*1000).toPrecision(3);$('surface_cell_mm').value=pm;}
  setDomainBoxes(r);
  const n=r.background_cells,count=n[0]*n[1]*n[2],f=v=>v.toFixed(3),max=+$('max_cells').value||5e6;
  /* cell sizes beside the resolution fields: each refinement level halves the background cell */
  chips($('cellinfo'),[['Background cell',mm(r.cell_size_m)],
    ['Background cells',n.join(' &times; ')+' = '+count.toLocaleString()],
    ['Surface cells',mm(r.surface_cell)+' (level '+r.level+')'],
    ['Wake cells',r.wake_cell?mm(r.wake_cell)+' (level '+$('wake_level').value+')':null],
    ['Of max cells',(100*count/max).toPrecision(2)+' % of '+max.toLocaleString()+' (before refinement)']]);
  const want=+$('surface_cell_mm').value;
  $('cellnote').textContent=want&&Math.abs(r.surface_cell*1000-want)>1e-6*want?
    want+' mm asked: the surface cells are the nearest finer size a whole level reaches.':'';
  chips($('livepreview'),[['Domain min (m)',r.domain_min.map(f).join(', ')],['Domain max (m)',r.domain_max.map(f).join(', ')],
    ['Domain size (m)',r.domain_max.map((v,i)=>(v-r.domain_min[i]).toPrecision(3)).join(' &times; ')]]);
  chips($('yplusinfo'),r.first_layer?[['First layer for y+ '+esc($('yplus').value),mm(r.first_layer)],
    ['Suggested layers',r.layers_suggest+' (to about half the '+mm(r.surface_cell)+' surface cells)']]:[],
    'Set a flow speed and viscosity to estimate the first layer height.');
  const t=r.transient,sec=v=>+v.toPrecision(3)+' s';
  if(t)$('end_time_s').placeholder='auto: '+sec(t.end);
  chips($('timeinfo'),t?[['Flow-through time',sec(t.flow_through)],['End time',sec(t.end)+' ('+(+(t.end/t.flow_through).toPrecision(2))+' flow-throughs)'],
    ['Field writes every',sec(t.write)],['First time step',sec(t.dt)],['Time steps','about '+big(Math.round(t.steps))+' at the first step']]:[]);
  const pr=r.probes,mmv=c=>c.map(v=>+(v*1000).toPrecision(4)).join(', ');
  const out=pr?pr.points.concat(pr.lines).filter(q=>!q.inside).map(q=>q.name):[];
  chips($('probeinfo'),pr?pr.points.map(q=>[q.name+(q.inside?'':' (outside)'),mmv(q.at)+' mm'])
    .concat(pr.lines.map(l=>[l.name.replace('line','Line ')+(l.inside?'':' (outside)'),mmv(l.start)+' → '+mmv(l.end)+' mm · '+pr.npts+' points'])):[],
    'No probes. They also show as markers in the domain preview.');
  setAlert($('probewarn'),'bad',out.length?esc(out.join(', '))+(out.length>1?' lie':' lies')+' outside the flow domain: Create case files will refuse '+(out.length>1?'them':'it')+'.':'');
  renderPhysics();
  renderSteps();
}
/* the aircraft mass in kg, from the number and its g / kg unit */
const massKg=()=>(+fv('mass')||0)*($('mass_unit').value==='g'?0.001:1);
/* the Physics card's effects: flow figures from the last preview, aircraft figures, and setting warnings */
const TURB_LABEL={nuTilda:'Inlet &nu;&#771; (m&sup2;/s)',k:'Inlet k (m&sup2;/s&sup2;)',omega:'Inlet &omega; (1/s)',
  ReThetat:'Inlet Re<sub>&theta;t</sub>',gammaInt:'Inlet &gamma;'};
function renderPhysics(r=LAST_PREVIEW){
  const p3=v=>+v.toPrecision(3);
  if(r)chips($('physstats'),[['Reynolds',Number(p3(r.Re)).toLocaleString()],
    ['Dynamic pressure q',p3(r.q)+' Pa'],
    ['Mach',$('fluid').value==='air'?p3(+$('speed').value/343)+(+$('speed').value/343>0.3?' (too fast: the solver is incompressible)':' (incompressible: fine)'):null],
    ['Flow direction',r.flow.map(v=>+v.toFixed(3)).join(', ')],
    ['A<sub>ref</sub>',p3(r.Aref)+' m&sup2;'],['l<sub>ref</sub>',p3(r.lRef)+' m'],
    ['Span',p3(r.span)+' m'+(fv('half_model')!=='none'?' (both halves)':'')],
    ['Moment ref. (m)',r.CofR.map(v=>+v.toPrecision(4)).join(', ')],
    ...Object.entries(r.inlet_turb||{}).map(([k,v])=>[TURB_LABEL[k]||esc(k),String(p3(v))])]);
  /* aircraft: wing loading needs the wing area (Aircraft reference values); the level-flight speed needs the
     simulated Cl, so it comes from the case's history (computed server-side by history()) */
  const mass=massKg(),wing=fv('ref_mode')==='custom',fl=HIST&&HIST.flight;
  chips($('flightstats'),mass>0&&wing&&r?[['Wing loading',p3(mass*9.80665/r.Aref)+' N/m&sup2; · '+p3(mass*1000/(r.Aref*100))+' g/dm&sup2;'],
    ['Level-flight speed',fl?(fl.v_level?p3(fl.v_level)+' m/s at the simulated Cl':'none (Cl &le; 0)'):null]]:[]);
  $('flightnote').textContent=!(mass>0)?'Enter an aircraft mass for the wing loading and flight speed.':
    !wing?'Select Aircraft reference values (wing area) for the wing loading and flight speed.':
    !fl?'Run the solver for the level-flight speed at the simulated Cl.':'';
  /* conflicts that will fail are .bad; the automatic freestream switch is only worth knowing (.info) */
  const bad=[];
  if(+fv('aoa')&&fv('ground')!=='none')bad.push('A ground can\'t be combined with an angle of attack.');
  if(+fv('yaw')&&fv('half_model')!=='none')bad.push('A half model can\'t be combined with yaw.');
  if(bad.length)setAlert($('physwarn'),'bad',bad.join(' '));
  else setAlert($('physwarn'),'info',+fv('aoa')||+fv('yaw')?'The inflow is tilted, so the side walls are Freestream (far field).':'');
}
function useSuggestedLayers(){
  if(!LAST_PREVIEW||!LAST_PREVIEW.layers_suggest){alert('Load an STL and set the flow first.');return;}
  $('layers').value=LAST_PREVIEW.layers_suggest;$('layer_mode').value='yplus';
}
async function updatePreview(){
  showStats(await j('/api/preview',{method:'POST',body:JSON.stringify(params())}));
}
['scale','dir','up_axis','aoa','yaw','ground','ground_gap_mm','pad_side','pad_up','pad_down','cell_size_mm','max_cells',
 'speed','rho','nu','wake_level','wake_len','yplus','expansion','half_model','aref','lref','cofr_x','cofr_y','cofr_z',
 'turb_model','nut_ratio','turb_intensity','omega_mode','turb_length_mm','sim_type',
 'end_time_s','write_count','max_co','delta_t','probe_points','probe_lines','probe_npts']
 .forEach(id=>$(id).addEventListener('change',updatePreview));

/* data-when="field=v1 v2": shown only while that select has one of those values (hidden fields keep
   their values). Used for the mesh resolution mode and the domain size mode */
function syncWhen(){
  /* "field=v1 v2" shows el while the select has one of those values; several conditions joined by ';' must all hold */
  document.querySelectorAll('[data-when]').forEach(el=>{
    el.style.display=el.dataset.when.split(';').every(c=>{const [f,vals]=c.split('=');return vals.split(' ').includes($(f).value);})?'':'none';
  });
}
/* resolution: a preset fills the two cell-size fields (showStats, from the preview); editing either field
   makes it Custom ('surface'), which keeps the shown sizes. Surface sizes mirror MESH_PRESETS */
const PRESET_MM={surf4:4,surf2:2,surf1:1,surf05:0.5};
$('mesh_preset').addEventListener('change',updatePreview);
['cell_size_mm','surface_cell_mm'].forEach(id=>$(id).addEventListener('input',()=>{$('mesh_preset').value='surface';}));
$('domain_mode').addEventListener('change',()=>{
  const r=LAST_PREVIEW;
  if($('domain_mode').value==='absolute'&&r){
    /* a centre/custom half model's span-axis min is the cut plane (the server overrides it while the half model
       is on): seed the full domain mirrored about it instead, so switching the half model off still encloses the part */
    const lo=r.domain_min.slice(),sa='xyz'.indexOf(spanAxis($('dir').value[1]));
    if(['centre','custom'].includes(fv('half_model')))lo[sa]=2*lo[sa]-r.domain_max[sa];
    'xyz'.split('').forEach((c,i)=>{$('dom_'+c+'0').value=+lo[i].toPrecision(5);$('dom_'+c+'1').value=+r.domain_max[i].toPrecision(5);});
  }
  syncWhen();updatePreview();
});
['surface_cell_mm','dom_x0','dom_x1','dom_y0','dom_y1','dom_z0','dom_z1'].forEach(id=>$(id).addEventListener('change',updatePreview));
$('ref_mode').addEventListener('change',()=>{
  const r=LAST_PREVIEW;
  if($('ref_mode').value==='custom'&&r&&!$('aref').value){
    $('aref').value=+r.Aref.toPrecision(4);$('lref').value=+r.lRef.toPrecision(4);
    'xyz'.split('').forEach((c,i)=>{$('cofr_'+c).value=+r.CofR[i].toPrecision(5);});
  }
  syncWhen();updatePreview();
});
['turb_model','omega_mode','sim_type','half_model'].forEach(id=>$(id).addEventListener('change',syncWhen));
/* a custom cut plane starts at the part's centre on the span axis (file units -> mm), ready to adjust */
$('half_model').addEventListener('change',()=>{
  if($('half_model').value!=='custom'||$('half_mm').value||!PART_BOX)return;
  const b=PART_BOX['xyz'.indexOf(spanAxis($('dir').value[1]))];
  $('half_mm').value=+((b[0]+b[1])/2*(+$('scale').value||1)*1000).toPrecision(5);updatePreview();
});
$('half_mm').addEventListener('change',updatePreview);
/* mass and target margin only change the derived flight/stability figures: re-fetch, nothing re-runs */
['mass','mass_unit'].forEach(id=>$(id).addEventListener('change',()=>{renderPhysics();refreshCaseView();refreshSweep();}));
$('target_sm').addEventListener('change',refreshSweep);
/* mass and target margin only feed derived figures, so they are saved into the open case straight away
   (everything else stays a draft until the case files are created again) */
['mass','mass_unit','target_sm','pv_near_mm'].forEach(id=>$(id).addEventListener('change',()=>{
  if(!CUR)return;
  const o={mass:$('mass').value,mass_unit:$('mass_unit').value,target_sm:$('target_sm').value,pv_near_mm:$('pv_near_mm').value};
  j('/api/case_params',{method:'POST',body:JSON.stringify({case:CUR,params:o})});
}));

/* Basic / Intermediate / Advanced: a per-viewer preference, so localStorage is enough.
   body.mid = Intermediate or above (shows .mid-only), body.adv = Advanced (also shows .adv-only) */
const UI_MODES=['basic','mid','adv'];
function setUiMode(m){
  if(!UI_MODES.includes(m))m='basic';
  document.body.classList.toggle('mid',m!=='basic');
  document.body.classList.toggle('adv',m==='adv');
  $('uimode').querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
  try{localStorage.setItem('pifoam.mode',m);}catch(e){}
}
$('uimode').querySelectorAll('button').forEach(b=>b.onclick=()=>setUiMode(b.dataset.m));
try{setUiMode(localStorage.getItem('pifoam.mode'));}catch(e){setUiMode('basic');}

/* ---------- hover help: any [data-tip] element explains itself in one floating .tip ---------- */
const TIP=document.createElement('div');TIP.className='tip';TIP.setAttribute('aria-hidden','true');document.body.appendChild(TIP);
let tipFor=null,tipTimer=0;
function tipTo(el){
  if(el===tipFor)return;
  tipFor=el;clearTimeout(tipTimer);TIP.classList.remove('on');
  if(!el)return;
  tipTimer=setTimeout(()=>{   /* below the element (above it near the bottom edge), clamped to the viewport */
    TIP.textContent=el.dataset.tip;
    const r=el.getBoundingClientRect(),t=TIP.getBoundingClientRect(),m=8;
    TIP.style.left=Math.max(m,Math.min(r.left,innerWidth-t.width-m))+'px';
    TIP.style.top=(r.bottom+6+t.height+m>innerHeight?r.top-6-t.height:r.bottom+6)+'px';
    TIP.classList.add('on');
  },300);
}
document.addEventListener('pointerover',e=>tipTo(e.target.closest('[data-tip]')));
document.documentElement.addEventListener('pointerleave',()=>tipTo(null));
document.addEventListener('focusin',e=>tipTo(e.target.closest('[data-tip]')));
document.addEventListener('focusout',()=>tipTo(null));
document.addEventListener('scroll',()=>tipTo(null),true);
document.addEventListener('keydown',e=>{if(e.key==='Escape')tipTo(null);});
/* screen readers get the same text as the control's description, also for rows rendered later */
function ariaTips(){
  document.querySelectorAll('[data-tip]').forEach(el=>{
    const c=el.matches('button,input,select')?el:el.querySelector('input,select,button');
    if(c&&c.getAttribute('aria-description')!==el.dataset.tip)c.setAttribute('aria-description',el.dataset.tip);
  });
}
ariaTips();new MutationObserver(ariaTips).observe($('casecard'),{childList:true,subtree:true});

/* simulation types (defined server-side in SIM_TYPES) */
const SIM_TYPES=/*SIM_TYPES*/null;
$('sim_type').innerHTML=Object.entries(SIM_TYPES).map(([k,v])=>'<option value="'+k+'">'+esc(v)+'</option>').join('');
const simLabel=k=>SIM_TYPES[k]||k||'';
/* RAS turbulence models (defined server-side in TURB_MODELS) */
const TURB_MODELS=/*TURB_MODELS*/null;
$('turb_model').innerHTML=Object.entries(TURB_MODELS).map(([k,v])=>'<option value="'+k+'">'+esc(v)+'</option>').join('');

/* feature panels: each .feature[data-feature] gets its heading (the FEATURE_GROUPS label) and an On / Off
   .seg[data-fseg]; the Studies card carries its seg in the card heading. Switching only updates #features:
   the fields keep their values, fv() and apply_features() ignore them while the feature is off */
document.querySelectorAll('.feature[data-feature]').forEach(el=>{
  el.querySelector('.feature-h').innerHTML='<span>'+esc(FEATURE_GROUPS[el.dataset.feature][0])+'</span><span class="spacer"></span>'+
    '<div class="seg" data-fseg="'+el.dataset.feature+'"></div>';
});
document.querySelectorAll('[data-fseg]').forEach(seg=>{
  const k=seg.dataset.fseg,label=esc(FEATURE_GROUPS[k][0].toLowerCase());
  segPicker(seg,[['on','On','Use '+label+' in this case.'],
    ['off','Off','Switch '+label+' off for this case: the settings are kept, but not used until it is on again.']],v=>{
    const off=new Set(featOff());if(v==='on')off.delete(k);else off.add(k);
    $('features').value=[...off].join(' ');
    syncFeatures();updatePreview();saveDraft();renderSteps();
    if(k==='compare'&&LAST_HIST){drawChart(LAST_HIST);renderCompare();}
  });
});
function syncFeatures(){
  const off=featOff();
  document.querySelectorAll('[data-fseg]').forEach(seg=>seg.querySelectorAll('button').forEach(b=>
    b.classList.toggle('on',(b.dataset.k==='off')===off.includes(seg.dataset.fseg))));
  document.querySelectorAll('.feature[data-feature]').forEach(el=>el.classList.toggle('off',off.includes(el.dataset.feature)));
  document.querySelectorAll('[data-feat]').forEach(el=>el.classList.toggle('feat-off',off.includes(el.dataset.feat)));
  syncWhen();VIEWERS.forEach(x=>x.redraw());
}

/* fluid material library (defined server-side in MATERIALS) */
const MATERIALS=/*MATERIALS*/null;
$('fluid').innerHTML=Object.entries(MATERIALS).map(([k,m])=>'<option value="'+k+'">'+m[0]+'</option>').join('')+'<option value="custom">Custom fluid</option>';
$('fluid').onchange=()=>{const m=MATERIALS[$('fluid').value];if(m){$('nu').value=m[1];$('rho').value=m[2];updatePreview();}};
['rho','nu'].forEach(id=>$(id).addEventListener('input',()=>{$('fluid').value='custom';}));
$('nprocs').value=/*NPROCS*/1;
const CORES=/*CORES*/[1,1];   /* [physical cores, hardware threads] */
$('nprocslabel').textContent='Processes ('+CORES[0]+' physical cores, '+CORES[1]+' threads; 1 = serial)';
$('file').onchange=e=>uploadFile(e.target.files[0]);
const dz=$('dropzone');
dz.ondragover=e=>{e.preventDefault();dz.classList.add('drag');};
dz.ondragleave=()=>dz.classList.remove('drag');
dz.ondrop=e=>{e.preventDefault();dz.classList.remove('drag');uploadFile(e.dataTransfer.files[0]);};

/* ---------- setup / run ---------- */
const PARAMS=['case','sim_type','scale','fluid','speed','dir','up_axis','aoa','yaw','rho','nu','ground','ground_gap_mm',
  'domain_mode','pad_side','pad_up','pad_down','dom_x0','dom_x1','dom_y0','dom_y1','dom_z0','dom_z1',
  'mesh_preset','cell_size_mm','surface_cell_mm','endtime',
  'layers','expansion','layer_mode','yplus','wake_level','wake_len','nut_ratio','side_bc','n_between','resolve_angle','feature_angle',
  'max_cells','final_layer','min_thick','write_interval','relax_U','relax_p','relax_nuTilda','res_p','res_U',
  'half_model','ref_mode','aref','lref','cofr_x','cofr_y','cofr_z','mass','mass_unit','turb_model','turb_intensity','target_sm','pv_near_mm',
  'omega_mode','turb_length_mm',
  'end_time_s','write_count','max_co','init_case','delta_t','time_scheme','n_outer','n_corr',
  'probe_points','probe_lines','probe_npts','half_mm','features'];
function params(){const o={};PARAMS.forEach(k=>o[k]=$(k).value);return o;}
function meshCfg(){return {auto_retry:$('auto_retry').checked,max_retries:+$('max_retries').value,level_bump:+$('level_bump').value};}
function autostopCfg(){return {enabled:$('autostop_enabled').checked,tol:+$('as_tol').value,window:+$('as_window').value,consecutive:+$('as_consec').value,min_iter:+$('as_miniter').value,poll_s:+$('as_poll').value};}
async function setup(skipConfirm){
  if(!CUR){alert('Choose or create a case first.');return false;}
  if(skipConfirm!==true&&!confirm('This (re)creates the case folder and deletes any old results in it. Continue?'))return false;
  const p=params(),r=await j('/api/setup',{method:'POST',body:JSON.stringify(p)});
  $('setupinfo').textContent=r.ok?('Created '+r.path):('Error: '+r.error);
  if(r.ok){WRITTEN=p;clearDraft(CUR);showStats(r);refreshCaseView();}
  await loadCases();
  return !!r.ok;
}
async function act(a,view){
  if(['mesh','solve','all'].includes(a)&&changedKeys().length){   /* the written case no longer matches the form */
    if(!confirm(changedKeys().length+' setting(s) changed since the case files were created, so this runs the written case, not the current settings.\n\nOK: run the written case anyway.\nCancel: more options.')){
      if(!confirm('Create the case files again from the current settings first? This deletes the case\'s mesh and results.')||!await setup(true))return;
    }
  }
  const r=await j('/api/run',{method:'POST',body:JSON.stringify({action:a,case:CUR,mesh:meshCfg(),autostop:autostopCfg(),nprocs:+$('nprocs').value,view,
    near_mm:$('pv_near_mm').value,probe_lines:$('probe_lines').value,probe_npts:$('probe_npts').value})});
  if(!r.ok)alert(r.error);
  else if(a==='sample'&&WRITTEN){WRITTEN.probe_lines=$('probe_lines').value;WRITTEN.probe_npts=$('probe_npts').value;renderSteps();}
}
/* ParaView presets (PV_VIEWS server-side): one button each */
const PV=[['overview','Overview','Domain outline, Cp on the part'],['uslice','Velocity slice','|U| on a plane through the part'],
  ['pslice','Pressure slice','Cp on the same plane'],['streamlines','Flow pathlines','Streamlines from upstream, coloured by |U|'],
  ['bodyp','Pressure on body','Cp on the part surface'],['wake','Wake','Surface where |U| is half the flow speed, coloured by Cp']];
$('pvbtns').innerHTML=PV.map(([v,t,d])=>'<button class="panel interactive" onclick="act(\'paraview\',\''+v+'\')" data-tip="Opens ParaView on this case with this view already set up, at the last saved iteration.">'+icon('external')+t+'<small>'+d+'</small></button>').join('');
ariaTips();

/* ---------- job queue ---------- */
function renderQueue(q){
  const el=$('queue');
  const key=(q||[]).map(x=>x.id).join(',');if(el.dataset.n===key)return;el.dataset.n=key;
  if(!key){el.innerHTML='';return;}
  el.innerHTML='<div class="caserow"><span class="name">Queued</span><span class="badge">'+q.length+' waiting</span>'+
    '<span class="spacer"></span><button class="sec sm" onclick="queueClear()">Clear queue</button></div>'+
    q.map((x,i)=>'<div class="caserow"><span>#'+(i+1)+'</span><span class="name">'+esc(x.case)+'</span><span class="badge">'+esc(x.kind)+'</span>'+
    '<span class="spacer"></span><button class="ghost sm" onclick="queueRemove('+x.id+')">Remove</button></div>').join('');
}
async function queueRemove(id){await j('/api/queue/remove',{method:'POST',body:JSON.stringify({id})});}
async function queueClear(){if(confirm('Remove all queued (not yet started) jobs?'))await j('/api/queue/clear',{method:'POST',body:'{}'});}
function esc(s){return String(s).replace(/[&<>"']/g,c=>'&#'+c.charCodeAt(0)+';');}

/* ---------- case manager ---------- */
/* case-first workflow: the workflow cards and step navigator only appear once a case (new or recent) is chosen */
let CASES=[],CUR=null,FOLDERS=[],FOLDER='';   /* FOLDER: the folder open on the home screen ('' = top level) */
const caseName=s=>String(s).trim().replace(/[^A-Za-z0-9_]/g,'_');   /* mirrors sanitize_name() */
/* a case (or folder) is 'case' at the top level or 'folder/sub/case', as case_rel() names it */
const folderOf=n=>n.includes('/')?n.slice(0,n.lastIndexOf('/')):'',baseOf=n=>n.split('/').pop();
const caseLabel=n=>n.split('/').join(' / ');
const caseBadges=c=>!c?badge('not created yet'):
  (c.has_mesh?badge('meshed','ok'):badge('no mesh'))+' '+(c.has_results?badge('solved','ok'):badge('not solved'));
/* drafts: every workspace field is kept per case in localStorage as it's edited, so settings survive a reload
   or switching cases before the case files exist (and edits made after). "Create case files" and deleting the
   case clear it. The uploaded STL is not part of a draft. */
const draftKey=n=>'pifoam.draft.'+n;
const draftFields=()=>Array.from($('workspace').querySelectorAll('input[id],select[id]')).filter(el=>el.type!=='file');
function saveDraft(){
  if(!CUR)return;
  const o={};draftFields().forEach(el=>o[el.id]=el.type==='checkbox'?el.checked:el.value);
  try{localStorage.setItem(draftKey(CUR),JSON.stringify(o));}catch(e){}
}
function loadDraft(name,fromCase){
  try{
    const o=JSON.parse(localStorage.getItem(draftKey(name))||'null');
    if(!o)return false;
    if(fromCase)delete o.scale;   /* a reopened case's part.stl is already in metres */
    /* a select only takes a value it still offers (drafts from older versions may hold retired options) */
    const ok=el=>el.tagName!=='SELECT'||Array.from(el.options).some(x=>x.value===String(o[el.id]));
    draftFields().forEach(el=>{if(el.id in o&&ok(el)){if(el.type==='checkbox')el.checked=o[el.id];else el.value=o[el.id];}});
    syncFeatures();return true;
  }catch(e){return false;}
}
function clearDraft(name){try{localStorage.removeItem(draftKey(name));}catch(e){}}
['input','change'].forEach(ev=>$('workspace').addEventListener(ev,()=>{saveDraft();renderSteps();}));
function setCurrent(name,fromCase){
  CUR=name;CMP=null;
  if(name)casePickers();   /* before the draft, so its warm-start / compare choices are offered */
  if(name&&loadDraft(name,fromCase)&&STLNAME)updatePreview();
  renderCurrent();
  if(name){
    VIEWERS.forEach(x=>x.redraw());refreshCaseView();
  }
}
function renderCurrent(){
  $('casepick').style.display=CUR?'none':'';
  $('curcase').style.display=CUR?'':'none';
  $('workspace').style.display=CUR?'':'none';
  $('steps').style.display=CUR?'':'none';setHdr();
  $('casehint').textContent=CUR?'Current case':'Open a recent case or start a new one';
  if(!CUR){
    $('tagline').textContent='Open a recent case or start a new one.';
    const n=$('case').value;   /* the band edits folder and name separately */
    if(n.includes('/')){setNewFolder(folderOf(n));$('case').value=baseOf(n);}
    return;
  }
  const c=CASES.find(x=>x.name===CUR);
  const type=simLabel(c?c.sim_type:$('sim_type').value);
  $('tagline').textContent=type+' · '+caseLabel(CUR);
  /* the title block: ruled label/value cells; Mesh, Cd and Status are filled by renderSteps */
  const cell=(cls,label,val)=>'<div'+(cls?' class="'+cls+'"':'')+'><b>'+label+'</b><span>'+val+'</span></div>';
  $('curcase').innerHTML=cell('tb-name','Case',esc(baseOf(CUR)))+cell('','Folder',esc(folderOf(CUR)?caseLabel(folderOf(CUR)):'Top level'))+
    cell('','Type',esc(type))+'<div><b>Mesh</b><span id="tb-mesh"></span></div><div><b>Cd</b><span id="tb-cd"></span></div>'+
    '<div><b>Status</b><span id="tb-state"></span></div>'+
    '<div class="tb-act"><button id="modrevert" class="ghost sm" onclick="revertCase()" data-tip="Put every changed setting back to what the written case files were created with.">Revert to written case</button>'+
    '<button class="sec sm" onclick="FOLDER=folderOf(CUR);setCurrent(null);loadCases()" data-tip="Back to the case list, in this case\'s folder.">Change case</button></div>';
  renderSteps();
}

/* ---------- step navigator: one header button per workflow card, with a live status line ---------- */
const STEPS=[['geomcard','Geometry'],['physcard','Physics'],['meshcard','Mesh'],['runcard','Run'],['rescard','Results'],['studycard','Studies']];
$('steps').innerHTML=STEPS.map(([id,t],i)=>'<button data-go="'+id+'"'+(id==='studycard'?' class="mid-only"':'')+
  ' data-tip="Jump to '+t+'."><span class="n">'+(i+1)+'</span>'+t+'<small id="st-'+id+'">&nbsp;</small></button>').join('');
$('steps').querySelectorAll('button').forEach(b=>b.onclick=()=>{
  const top=$(b.dataset.go).getBoundingClientRect().top+scrollY-$('top').offsetHeight;
  scrollTo({top,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});
});
/* --hdr: the sticky header's height, so each .split's sticky effects column parks just below it */
function setHdr(){document.documentElement.style.setProperty('--hdr',$('top').offsetHeight+'px');}
addEventListener('resize',setHdr);
/* the step whose card's top is nearest above the reading line (just under the header) is .on */
function markStep(){
  const line=$('top').offsetHeight+40;let cur=STEPS[0][0];
  for(const [id] of STEPS){const el=$(id);if(el.offsetParent&&el.getBoundingClientRect().top<=line)cur=id;}
  $('steps').querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.go===cur));
}
addEventListener('scroll',markStep,{passive:true});
let MESH=null,HIST=null,JOBSTATE=null,SWEEP=null;   /* latest server state behind the step status lines */
const big=n=>n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':String(n);
function renderSteps(){
  if(!CUR)return;
  const set=(id,t,ic)=>{$('st-'+id).innerHTML=(ic?icon(ic,'sm'):'')+esc(t);};
  const c=CASES.find(x=>x.name===CUR)||{},running=JOBSTATE&&JOBSTATE.case===CUR?JOBSTATE:null;
  set('geomcard',LAST_PREVIEW&&STLNAME?STLNAME:'No STL loaded');
  const fl=$('fluid'),fluid=fl.value==='custom'?'custom fluid':fl.options[fl.selectedIndex].text;
  const tm=$('turb_model');
  const tilt=[+fv('aoa')&&'AoA '+fv('aoa')+'°',+fv('yaw')&&'yaw '+fv('yaw')+'°',fv('ground')!=='none'&&'ground',
    fv('half_model')!=='none'&&'half model',tm.selectedIndex>0&&tm.options[tm.selectedIndex].text].filter(Boolean);
  set('physcard',[$('speed').value+' m/s',fluid].concat(LAST_PREVIEW?['Re '+big(+LAST_PREVIEW.Re.toPrecision(2))]:[],tilt).join(' · '));
  const st=MESH&&MESH.stats||{};
  set('meshcard',running&&/mesh|all/.test(running.kind)&&running.meta.step?'Running '+running.meta.step:
    MESH&&MESH.last_result?(st.cells?big(st.cells)+' cells':MESH.last_result.replace(/_/g,' ')):c.has_mesh?'meshed':'Not meshed',
    !(running&&/mesh|all/.test(running.kind)&&running.meta.step)&&(MESH&&MESH.last_result?(MESH.last_result==='ok'?'check':'x'):c.has_mesh?'check':''));
  const pr=running&&progress(running.meta);
  set('runcard',pr?'Solving '+Math.round(pr.frac*100)+'%'+(pr.left!=null?' · '+dur(pr.left)+' left':''):
    running?'Running '+running.kind:c.has_results?'Solved':'Not solved',!pr&&!running&&c.has_results&&'check');
  const av=HIST&&HIST.averages&&HIST.averages.coeffs.Cd;
  set('rescard',av?'Cd '+av.mean.toFixed(4):'No results yet');
  /* the title block's live cells */
  const ok=MESH&&MESH.last_result?MESH.last_result==='ok':c.has_mesh?true:null;
  if($('tb-mesh')){
    $('tb-mesh').innerHTML=(ok==null?'':icon(ok?'check':'x','sm'))+esc(st.cells?big(st.cells)+' cells':ok==null?'none':ok?'meshed':'failed');
    $('tb-cd').textContent=av?av.mean.toFixed(4):c.Cd!=null?c.Cd.toFixed(4):'not solved';
  }
  set('studycard',SWEEP&&SWEEP.running?'Variant '+SWEEP.sweep_index+'/'+SWEEP.sweep_total:
    SWEEP&&SWEEP.results?SWEEP.results.length+' variants':'No sweep yet');
  const ch=changedKeys(),cards=[...new Set(ch.map(k=>$(k).closest('.card')?.id))].filter(Boolean);
  cards.forEach(id=>{$('st-'+id).insertAdjacentText('beforeend',' · edited');});
  renderModified(ch,cards);
  markStep();
}
/* The written case: the form settings its case files were created with (null for a case not written yet).
   Everything below compares the form against it. Mass and target margin are saved to the case live, so they
   are not differences. */
let WRITTEN=null;
const DIFF_SKIP=['case','sim_type','mass','mass_unit','target_sm','pv_near_mm'];
const writtenVal=k=>{const v=WRITTEN[k];return v!=null?String(v):k==='mesh_preset'?'surface':defaultOf($(k));};
/* with a preset, the cell-size fields are its output (rounded for display), not settings of their own */
const changedKeys=()=>WRITTEN?PARAMS.filter(k=>!DIFF_SKIP.includes(k)&&
  !(PRESET_MM[$('mesh_preset').value]&&(k==='cell_size_mm'||k==='surface_cell_mm'))&&$(k).value!==writtenVal(k)):[];
/* the 'modified' badge beside the case name, the Run card banner naming the cards with edits, and the revert buttons */
function renderModified(ch,cards){
  const on=!!CUR&&ch.length>0,show=(el,v)=>{if(el)el.style.display=v?'':'none';};
  show($('modrevert'),on);show($('modbanner'),on);
  const c=CUR&&CASES.find(x=>x.name===CUR),state=$('tb-state');
  if(state){state.textContent=!c?'not written':on?'modified':'written';
    if(on)state.dataset.tip='Some settings differ from the case files that were written. Running uses the written files until you create them again.';
    else delete state.dataset.tip;}
  if(on)$('modtext').textContent=ch.length+' setting'+(ch.length>1?'s':'')+' changed since the case files were created ('+
    cards.map(id=>STEPS.find(s=>s[0]===id)[1]).join(', ')+'). Mesh and Solve use the written case until you create the case files again.';
}
function revertCase(){
  const ch=changedKeys();
  if(!ch.length||!confirm('Revert '+ch.length+' changed setting'+(ch.length>1?'s':'')+' to the written case? Your edits are lost.'))return;
  ch.forEach(k=>{$(k).value=writtenVal(k);});
  syncFeatures();clearDraft(CUR);VIEWERS.forEach(x=>x.redraw());
  if(STLNAME)updatePreview();
  renderSteps();
}
/* solver progress from /api/state job meta: fraction of iterations done and seconds left */
function progress(m){
  if(!m||m.phase!=='solve'||m.iter==null||!m.end)return null;
  const frac=Math.min(1,m.iter/m.end),done=m.iter-m.iter0,el=(JOBSTATE.now-m.t0);
  /* sofar: "230 / 500" iterations, or simulated seconds for a transient run */
  const s=m.unit==='s',f=v=>s?String(+v.toPrecision(3)):String(Math.round(v));
  return {frac,iter:m.iter,end:m.end,left:done>0&&el>0?(m.end-m.iter)*el/done:null,
    sofar:f(m.iter)+' / '+f(m.end)+(s?' s':''),unit:s?'s simulated':'iterations'};
}
const dur=s=>s<60?Math.round(s)+'s':s<3600?Math.floor(s/60)+'m '+Math.round(s%60)+'s':Math.floor(s/3600)+'h '+Math.round(s%3600/60)+'m';
$('workspace').addEventListener('change',renderSteps);   /* e.g. speed/fluid edits update the Physics line */
ariaTips();
async function startCase(){
  const base=caseName(baseOf($('case').value)),folder=$('newfolder').value;
  if(!base.replace(/_/g,'')){alert('Enter a case name.');return;}
  const name=(folder?folder+'/':'')+base;
  await loadCases();
  if(CASES.some(c=>c.name===name)){
    if(confirm('A case named "'+caseLabel(name)+'" already exists. Open it instead?'))pickCase(name);
    return;
  }
  $('case').value=name;WRITTEN=null;setCurrent(name);
  const f=$('newfile').files[0];
  if(f){await uploadFile(f);$('newfile').value='';CASE_NAMED=false;}
  else if(sharedPart(folder)){   /* the folder's shared part (its own or an enclosing folder's) */
    const r=await j('/api/folders/use_stl',{method:'POST',body:JSON.stringify({name:folder})});
    if(r.ok)applyStl(r,r.name,r.scale);else alert(r.error);
  }
}
async function loadCases(){
  const r=await j('/api/cases');
  CASES=r.ok?r.cases:[];FOLDERS=r.ok?r.folders:[];
  if(FOLDER&&!FOLDERS.some(f=>f.name===FOLDER))FOLDER='';   /* it was renamed or deleted elsewhere */
  if(CUR)casePickers();
  renderFolder();renderCurrent();renderCaseGrid();
}
/* ---------- folders: directories of cases (and folders) in $FOAM_RUN, browsed like a file manager ---------- */
const folderInfo=n=>FOLDERS.find(f=>f.name===n);
/* the shared part that applies in a folder: its own, else the nearest enclosing folder's (mirrors folder_stl()) */
function sharedPart(n){for(;n;n=folderOf(n)){const f=folderInfo(n);if(f&&f.has_stl)return f;}return null;}
function setNewFolder(n){if(Array.from($('newfolder').options).some(o=>o.value===n))$('newfolder').value=n;}
function openFolder(n){
  FOLDER=n||'';$('casesearch').value='';
  try{localStorage.setItem('pifoam.folder',FOLDER);}catch(e){}
  setNewFolder(FOLDER);renderFolder();renderCaseGrid();
}
try{FOLDER=localStorage.getItem('pifoam.folder')||'';}catch(e){}
/* breadcrumb, the New case band's folder list and the open folder's row (shared part, rename, delete) */
function renderFolder(){
  const parts=FOLDER?FOLDER.split('/'):[];
  $('crumbs').innerHTML='<button data-drop="" onclick="openFolder(\'\')"'+(parts.length?'':' aria-current="page"')+
    ' data-tip="The top level of $FOAM_RUN. Drop a case or folder here to move it out of its folder.">All cases</button>'+
    parts.map((p,i)=>{const path=parts.slice(0,i+1).join('/');return '<span class="sep" aria-hidden="true">/</span><button data-drop="'+esc(path)+'"'+
      ' onclick=\'openFolder('+esc(JSON.stringify(path))+')\''+(i===parts.length-1?' aria-current="page"':'')+'>'+esc(p)+'</button>';}).join('');
  $('crumbs').querySelectorAll('button').forEach(dropTarget);
  const nf=$('newfolder');   /* new cases go in the open folder unless another one is picked */
  nf.innerHTML='<option value="">Top level</option>'+FOLDERS.map(f=>'<option value="'+esc(f.name)+'">'+esc(caseLabel(f.name))+'</option>').join('');
  nf.value=FOLDER;
  const f=folderInfo(FOLDER),row=$('folderrow'),sp=sharedPart(FOLDER);
  row.style.display=f?'':'none';
  if(!f)return;
  const a=esc(JSON.stringify(f.name));
  row.innerHTML=badge(f.all_cases+' case'+(f.all_cases===1?'':'s')+(f.folders?' with subfolders':''))+
    (f.has_stl?badge('shared part: '+f.stl,'ok'):sp?badge('shared part from '+caseLabel(sp.name)+': '+sp.stl,'ok'):badge('no shared part'))+'<span class="spacer"></span>'+
    '<div class="dropzone sm" id="folderdrop" data-tip="This folder\'s shared part. New cases in this folder and its subfolders start from it when you don\'t choose a file of their own (a subfolder\'s own part wins)."><input type="file" id="folderfile" accept=".stl" aria-label="Folder shared part (STL)"></div>'+
    '<select id="folderscale" class="sm" aria-label="Shared part units" data-tip="The units the shared STL was exported in. New cases take them over.">'+
      [['0.001','millimetres'],['1','metres'],['0.01','centimetres'],['0.0254','inches']].map(([v,t])=>'<option value="'+v+'"'+(v===f.scale?' selected':'')+'>'+t+'</option>').join('')+'</select>'+
    '<button class="ghost sm" onclick=\'renameFolder('+a+')\' data-tip="Rename this folder. Its cases and subfolders keep their names inside it.">Rename</button>'+
    '<button class="ghost sm" onclick=\'moveItem("folder",'+a+')\' data-tip="Move this folder into another folder (you can also drag its tile onto a folder or the breadcrumb).">Move…</button>'+
    '<button class="ghost sm danger" onclick=\'deleteFolder('+a+')\' data-tip="Delete this folder and its subfolders. If they still hold cases you are asked again before those are deleted too.">Delete folder</button>';
  $('folderfile').onchange=()=>setFolderStl(f.name,$('folderfile').files[0]);
  $('folderscale').onchange=()=>{const file=$('folderfile').files[0];if(file)setFolderStl(f.name,file);};
  const d=$('folderdrop');
  d.addEventListener('dragover',e=>{if(e.dataTransfer.types.includes('Files')){e.preventDefault();d.classList.add('drag');}});
  d.addEventListener('dragleave',()=>d.classList.remove('drag'));
  d.addEventListener('drop',e=>{e.preventDefault();d.classList.remove('drag');if(e.dataTransfer.files.length)setFolderStl(f.name,e.dataTransfer.files[0]);});
  ariaTips();
}
async function setFolderStl(name,f){
  if(!f)return;
  const r=await j('/api/folders/stl?folder='+encodeURIComponent(name)+'&name='+encodeURIComponent(f.name)+'&scale='+$('folderscale').value,
    {method:'POST',body:await f.arrayBuffer()});
  if(!r.ok)alert(r.error);
  Object.keys(THUMBS).filter(k=>k.startsWith('folder:')).forEach(k=>delete THUMBS[k]);
  loadCases();
}
async function newFolder(){
  const n=prompt('New folder'+(FOLDER?' in "'+caseLabel(FOLDER)+'"':'')+' (letters, digits and underscores):');if(!n)return;
  const r=await j('/api/folders/create',{method:'POST',body:JSON.stringify({name:n,parent:FOLDER})});
  if(!r.ok){alert(r.error);return;}
  await loadCases();
}
/* drafts and thumbnails are kept under the case's name: carry them over when names change (moves, renames) */
function renameDraft(from,to){
  try{const v=localStorage.getItem(draftKey(from));if(v!=null){localStorage.setItem(draftKey(to),v);localStorage.removeItem(draftKey(from));}}catch(e){}
  delete THUMBS[from];
}
/* a folder that moved from `from` to `to`: every case below it, and the open folder if it was inside */
function folderMoved(from,to){
  CASES.filter(c=>c.name.startsWith(from+'/')).forEach(c=>renameDraft(c.name,to+c.name.slice(from.length)));
  if(FOLDER===from||FOLDER.startsWith(from+'/'))FOLDER=to+FOLDER.slice(from.length);
  Object.keys(THUMBS).filter(k=>k.startsWith('folder:')).forEach(k=>delete THUMBS[k]);
}
async function renameFolder(name){
  const n=prompt('New name for folder "'+baseOf(name)+'":',baseOf(name));if(!n||n===baseOf(name))return;
  const r=await j('/api/folders/rename',{method:'POST',body:JSON.stringify({name,new:n})});
  if(!r.ok){alert(r.error);return;}
  folderMoved(name,r.name);openFolder(FOLDER);loadCases();
}
async function renameCase(name){
  const n=prompt('New name for case "'+baseOf(name)+'" (letters, digits and underscores):',baseOf(name));if(!n||n===baseOf(name))return;
  const r=await j('/api/cases/rename',{method:'POST',body:JSON.stringify({case:name,new:n})});
  if(!r.ok){alert(r.error);return;}
  renameDraft(name,r.name);loadCases();
}
async function deleteFolder(name){
  const inside=CASES.filter(c=>c.name.startsWith(name+'/')),n=inside.length;
  if(!confirm('Delete folder "'+caseLabel(name)+'"'+(n?' and the '+n+' case'+(n>1?'s':'')+' in it (with all their results)':'')+' permanently?'))return;
  const r=await j('/api/folders/delete',{method:'POST',body:JSON.stringify({name,with_cases:n>0})});
  if(!r.ok){alert(r.error);return;}
  inside.forEach(c=>clearDraft(c.name));
  openFolder(folderOf(name));loadCases();
}
/* move a case or a folder into a folder ('' = the top level): drag-and-drop, or Move… */
async function moveTo(kind,name,dest){
  if(dest===folderOf(name)||kind==='folder'&&(dest===name||dest.startsWith(name+'/')))return;
  const r=await j(kind==='case'?'/api/cases/move':'/api/folders/move',
    {method:'POST',body:JSON.stringify(kind==='case'?{case:name,folder:dest}:{name,dest})});
  if(!r.ok){alert(r.error);return;}
  if(kind==='case')renameDraft(name,r.name);else folderMoved(name,r.name);
  if(kind==='folder'&&FOLDER===r.name)openFolder(FOLDER);
  loadCases();
}
/* the keyboard way to move: type the destination folder */
function moveItem(kind,name){
  const d=prompt('Move "'+caseLabel(name)+'" to which folder? Give its path (e.g. '+(FOLDERS[0]?FOLDERS[0].name:'wing/tests')+'), or leave it blank for the top level.',folderOf(name));
  if(d==null)return;
  const dest=d.trim().replace(/^\/+|\/+$/g,'').split('/').map(s=>s.trim()).filter(Boolean).join('/');
  if(dest&&!folderInfo(dest)){alert('There is no folder "'+dest+'".');return;}
  moveTo(kind,name,dest);
}
/* drag-and-drop: case and folder tiles are dragged, folder tiles and breadcrumb entries take the drop */
const DND='application/x-pifoam';
function dragSource(el,kind,name){
  el.draggable=true;
  el.addEventListener('dragstart',e=>{e.dataTransfer.setData(DND,JSON.stringify({kind,name}));e.dataTransfer.effectAllowed='move';});
}
function dropTarget(el){
  const dest=el.dataset.drop;
  el.addEventListener('dragover',e=>{if(e.dataTransfer.types.includes(DND)){e.preventDefault();e.dataTransfer.dropEffect='move';el.classList.add('drop');}});
  el.addEventListener('dragleave',e=>{if(!el.contains(e.relatedTarget))el.classList.remove('drop');});
  el.addEventListener('drop',e=>{
    el.classList.remove('drop');const raw=e.dataTransfer.getData(DND);if(!raw)return;
    e.preventDefault();const {kind,name}=JSON.parse(raw);moveTo(kind,name,dest);
  });
}
/* the home screen: the open folder's subfolders and cases as tiles; a search looks through every folder */
let CASE_SORT='recent';
const ago=t=>{const s=Date.now()/1000-t;return s<60?'just now':s<3600?Math.floor(s/60)+' min ago':s<86400?Math.floor(s/3600)+' h ago':
  s<86400*14?Math.floor(s/86400)+' days ago':new Date(t*1000).toLocaleDateString();};
function renderCaseGrid(){
  const el=$('caselist'),q=$('casesearch').value.trim().toLowerCase();
  const folders=q?FOLDERS.filter(f=>f.name.toLowerCase().includes(q)):FOLDERS.filter(f=>f.parent===FOLDER);
  const list=q?CASES.filter(c=>c.name.toLowerCase().includes(q)):CASES.filter(c=>c.folder===FOLDER);
  const byName=(a,b)=>a.name.localeCompare(b.name);
  folders.sort(CASE_SORT==='name'?byName:(a,b)=>b.mtime-a.mtime);
  if(CASE_SORT==='name')list.sort(byName);
  if(!list.length&&!folders.length){el.className='note';el.textContent=q?'Nothing matches "'+$('casesearch').value+'".':
    FOLDER?'This folder is empty. Create a case above, a folder with New folder, or drag cases onto its name in the path.':'No cases yet. Create one above.';return;}
  el.className='casegrid';
  el.innerHTML=folders.map(f=>{
    const n=esc(f.name),a=esc(JSON.stringify(f.name)),sp=sharedPart(f.name);
    const what=[f.cases&&f.cases+' case'+(f.cases>1?'s':''),f.folders&&f.folders+' folder'+(f.folders>1?'s':'')].filter(Boolean).join(' · ')||'Empty';
    return '<div class="casetile panel interactive" data-folder="'+n+'" data-drop="'+n+'">'+
      '<button class="thumb" onclick=\'openFolder('+a+')\' aria-label="Open folder '+n+'" data-thumbfolder="'+n+'">'+
        (sp?'<canvas width="480" height="260"></canvas>':icon('folder','lg'))+'</button>'+
      '<div class="name" title="'+n+'">'+esc(baseOf(f.name))+'</div>'+
      '<div class="note">'+(q&&f.parent?esc(caseLabel(f.parent))+' · ':'')+'Folder · '+what+'</div>'+
      (sp?'<div>'+badge('shared part','ok')+'</div>':'')+
      '<div class="actions">'+
      '<button class="sec sm" onclick=\'openFolder('+a+')\' data-tip="Show the cases and folders inside. Drop a case or folder on this tile to move it in.">Open</button>'+
      '<button class="ghost sm" onclick=\'moveItem("folder",'+a+')\' data-tip="Move this folder into another folder; you can also drag the tile onto a folder or the path above.">Move…</button></div></div>';
  }).join('')+list.map(c=>{
    const n=esc(c.name),a=esc(JSON.stringify(c.name));
    const figs=[c.cells&&['Cells',big(c.cells)],c.Cd!=null&&['Cd',c.Cd.toFixed(4)],c.Cl!=null&&['Cl',c.Cl.toFixed(4)]].filter(Boolean);
    return '<div class="casetile panel interactive" data-case="'+n+'">'+
      '<button class="thumb" onclick=\'pickCase('+a+')\' aria-label="Open '+n+'" data-case="'+n+'">'+
        (c.has_stl?'<canvas width="480" height="260"></canvas>':'<span class="note">No geometry</span>')+'</button>'+
      '<div class="name" title="'+n+'">'+esc(baseOf(c.name))+'</div>'+
      '<div class="note">'+(q&&c.folder?esc(caseLabel(c.folder))+' · ':'')+ago(c.mtime)+' · '+esc(simLabel(c.sim_type))+'</div>'+
      '<div>'+caseBadges(c)+'</div>'+
      (figs.length?'<div class="stats">'+figs.map(([k,v])=>'<div class="stat"><b>'+k+'</b>'+v+'</div>').join('')+'</div>':'')+
      '<div class="actions">'+
      '<button class="sm" onclick=\'pickCase('+a+')\' data-tip="Reopen this case with its saved settings and geometry.">Open</button>'+
      '<button class="ghost sm" onclick=\'useTemplate('+a+')\' data-tip="Start a new case (named …_copy, renameable) in the same folder with this case\'s settings and geometry.">Use as template</button>'+
      '<button class="ghost sm" onclick=\'renameCase('+a+')\' data-tip="Give the case a new name, in the same folder. Its settings, mesh and results stay as they are.">Rename</button>'+
      '<button class="ghost sm" onclick=\'moveItem("case",'+a+')\' data-tip="Move the case into another folder; you can also drag its tile onto a folder or the path above.">Move…</button>'+
      '<span class="spacer"></span><button class="ghost sm danger" onclick=\'deleteCase('+a+')\' data-tip="Delete the case and all its results. This can\'t be undone.">Delete</button></div></div>';
  }).join('');
  el.querySelectorAll('.casetile[data-folder]').forEach(t=>{dragSource(t,'folder',t.dataset.folder);dropTarget(t);});
  el.querySelectorAll('.casetile[data-case]').forEach(t=>dragSource(t,'case',t.dataset.case));
  el.querySelectorAll('button.thumb[data-case] canvas').forEach(cv=>drawThumb(cv,cv.parentNode.dataset.case));
  el.querySelectorAll('button.thumb[data-thumbfolder] canvas').forEach(cv=>drawThumb(cv,cv.parentNode.dataset.thumbfolder,true));
}
const THUMBS={};   /* case name (or 'folder:' + folder) -> decimated part preview, fetched once per page load */
async function drawThumb(canvas,name,folder){
  const key=(folder?'folder:':'')+name;
  if(!THUMBS[key]){const r=await j('/api/thumb?'+(folder?'folder=':'case=')+encodeURIComponent(name));THUMBS[key]=r.ok?r.preview:[];}
  makeViewer(canvas,{static:true,zoom:1.4}).set(THUMBS[key]);
}
$('casesearch').addEventListener('input',renderCaseGrid);
segPicker($('casesort'),[['recent','Recent','Most recently changed first.'],['name','Name','Alphabetical.']],k=>{CASE_SORT=k;renderCaseGrid();});
/* the New case band's STL: its file name names the case while the name is untouched; uploaded on Create */
let CASE_NAMED=false;
$('case').addEventListener('input',()=>{CASE_NAMED=true;});
$('newfile').addEventListener('change',()=>{
  const f=$('newfile').files[0];if(!f)return;
  if(!CASE_NAMED)$('case').value=caseName(f.name.replace(/\.stl$/i,''));
});
const nd=$('newdrop');
nd.addEventListener('dragover',e=>{e.preventDefault();nd.classList.add('drag');});
nd.addEventListener('dragleave',()=>nd.classList.remove('drag'));
nd.addEventListener('drop',e=>{e.preventDefault();nd.classList.remove('drag');
  if(e.dataTransfer.files.length){$('newfile').files=e.dataTransfer.files;$('newfile').dispatchEvent(new Event('change'));}});
/* a field's value as written in the HTML (inputs: value attribute, selects: the selected option) */
function defaultOf(el){
  if(el.tagName!=='SELECT')return el.defaultValue;
  return (Array.from(el.options).find(o=>o.defaultSelected)||el.options[0]).value;
}
/* fill the form + geometry viewer from a case's saved settings and part.stl. Fields the case
   didn't save fall back to their defaults; cases from before mesh presets are Custom (the server converts
   their saved cell count / cells across part to cell_size_mm) */
async function fillFromCase(name,keep){
  const r=await j('/api/case_template',{method:'POST',body:JSON.stringify({case:name,keep_name:keep})});
  if(!r.ok){alert(r.error);return false;}
  casePicker($('init_case'),'<option value="">Uniform free stream</option>',r.params.init_case||'',c=>c.has_results);
  PARAMS.forEach(k=>{const v=r.params[k];$(k).value=v!=null?v:(k==='mesh_preset'?'surface':defaultOf($(k)));});
  syncFeatures();
  WRITTEN=keep&&r.params.speed!=null?r.params:null;   /* a case from before saved settings has nothing to compare with */
  if(r.preview){
    showFile('part.stl from case '+name+' · '+big(r.tris)+' triangles · size '+r.ext.map(v=>v.toPrecision(4)).join(' x ')+' m. Choose another file to replace it.',true);
    STLNAME='part.stl · '+big(r.tris)+' triangles';
    setPreview(r.preview);updatePreview();
  }else{
    STLNAME='';
  }
  return true;
}
/* "Load": reopen the case itself */
async function pickCase(name){
  if(!await fillFromCase(name,true))return;
  setCurrent(name,true);
}
/* "Use as template": start a new case (<name>_copy, renameable) from a case's settings + geometry */
async function useTemplate(name){
  if(!await fillFromCase(name,false))return;
  setCurrent(null);$('case').focus();$('case').select();
}
async function deleteCase(name){
  if(!confirm('Delete case "'+name+'" permanently?'))return;
  const r=await j('/api/cases/delete',{method:'POST',body:JSON.stringify({case:name})});
  if(!r.ok)alert(r.error);
  else{clearDraft(name);if(name===CUR)setCurrent(null);}
  loadCases();
}

/* ---------- mesh status / chart / table / results ---------- */
function renderMeshStatus(ms){
  MESH=ms;
  if(!ms||!ms.last_result){$('meshstatus').textContent='No mesh run yet.';setAlert($('meshwarn'),'warn','');$('meshstats').innerHTML='';$('meshstats').className='';return;}
  const ok=ms.last_result==='ok',label=ok?'Mesh OK':(ms.last_result==='failed'?'Mesh failed':ms.last_result.replace(/_/g,' '));
  const warn=ms.warnings||[];
  $('meshstatus').innerHTML=badge(label,ok?'ok':'bad')+' refinement level '+ms.level+', attempt '+ms.attempts+
    (ms.last_failed_checks?', '+ms.last_failed_checks+' failed checks':'');
  setAlert($('meshwarn'),ok?'warn':'bad',warn.map(esc).join('<br>'));
  /* checkMesh quality figures; a check checkMesh flagged gets a "failed" badge (text, not just colour), or
     "tolerated" when the mesh passed anyway (a few surface faces within snappyHexMesh's skewness limit) */
  const st=ms.stats||{};
  const flag=k=>!st[k].fail?'':ok&&k==='skewness'?' '+badge('tolerated'):' '+badge('failed','bad');
  const q=(k,label,fmt)=>st[k]?[[label,fmt(st[k])+flag(k)]]:[];
  const rows=[].concat(st.cells?[['Cells',st.cells.toLocaleString()]]:[],st.faces?[['Faces',st.faces.toLocaleString()]]:[],
    q('non_orth','Non-orthogonality',v=>v.max.toFixed(1)+'&deg; max, '+v.avg.toFixed(1)+'&deg; avg'),
    q('skewness','Max skewness',v=>v.max.toFixed(2)),q('aspect','Max aspect ratio',v=>v.max.toFixed(1)));
  $('meshstats').className=rows.length?'stats':'';
  $('meshstats').innerHTML=rows.map(([k,v])=>'<div class="stat"><b>'+k+'</b>'+v+'</div>').join('');
}
/* ---------- charts: one HiDPI line-plot renderer (gridlines, tick labels, title, legend) ---------- */
const SERIES_COL=['--c1','--c2','--c3','--c4','--c5','--c6'].map(tok);
function niceTicks(lo,hi,n){   /* ~n ticks on 1/2/5 x 10^k steps covering [lo,hi] */
  const raw=(hi-lo)/n,mag=Math.pow(10,Math.floor(Math.log10(raw))),f=raw/mag;
  const step=(f<1.5?1:f<3?2:f<7?5:10)*mag,t=[];
  for(let v=Math.floor(lo/step)*step;v<=hi+step*1e-6;v+=step)t.push(+v.toPrecision(12));
  return t;
}
/* plain-decimal numbers, never e-notation. With a tick step: just enough decimals for that step; without:
   4 significant digits. Large values get k / M */
const trimZeros=s=>s.includes('.')?s.replace(/\.?0+$/,''):s;
function fmtNum(v,step){
  const a=Math.abs(v),dec=s=>Math.max(0,Math.ceil(-Math.log10(s)-1e-9));
  for(const [d,u,from] of [[1e6,'M',1e6],[1e3,'k',1e4]])
    if(a>=from&&(step==null||step>=d/10))return trimZeros((v/d).toFixed(step==null?Math.max(0,3-Math.floor(Math.log10(a/d))):dec(step/d)))+u;
  if(step!=null)return (+v.toFixed(dec(step))).toFixed(dec(step));
  if(!a)return '0';
  return trimZeros(v.toFixed(Math.min(20,Math.max(0,3-Math.floor(Math.log10(a))))));
}
const fmtDecade=e=>e>=0?String(Math.pow(10,e)):'0.'+'0'.repeat(-e-1)+'1';   /* log axis label for 10^e */
const CHART_PH=180;   /* plot-area height in px, whatever the width */
/* series: [{name, xs, ys, color?}] (colour defaults to SERIES_COL in order);
   opts: {title, value?, log?, xlabel? (default 'Iteration'), dots? (mark each point: few-point charts),
   fitAll? (fit the y range to every point: profiles, where the start isn't a transient)}.
   Hovering shows a crosshair and a readout of each series at the nearest point. */
function plot(canvas,series,opts){
  canvas._plot=[series,opts];
  if(!canvas._hover){   /* hover listeners once per canvas; redraws are coalesced to one per frame */
    canvas._hover=true;
    const again=()=>{if(!canvas._raf)canvas._raf=requestAnimationFrame(()=>{canvas._raf=0;drawPlot(canvas);});};
    canvas.addEventListener('mousemove',e=>{canvas._hx=e.offsetX;again();});
    canvas.addEventListener('mouseleave',()=>{canvas._hx=null;again();});
  }
  drawPlot(canvas);
}
function drawPlot(canvas){
  if(!canvas.clientWidth||!canvas._plot)return;   /* hidden: drawn again on the next refresh */
  const [series,opts]=canvas._plot,col=k=>series[k].color||SERIES_COL[k%SERIES_COL.length];
  const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,ctx=canvas.getContext('2d');
  const css=n=>getComputedStyle(document.body).getPropertyValue(n).trim(),mut=css('--mut'),fg=css('--fg'),font=css('--font');
  const R=16,B=36,ph=CHART_PH;
  const size=T=>{   /* backing store for a T-high header; CSS height:auto follows this aspect. Resets ctx state */
    canvas.width=Math.round(w*dpr);canvas.height=Math.round((T+ph+B)*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
  };
  const title=L=>{
    ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.font='600 12px '+font;ctx.fillStyle=fg;ctx.fillText(opts.title,L,20);
    if(opts.value){const tw=ctx.measureText(opts.title).width;ctx.font='12px '+font;ctx.fillStyle=mut;ctx.fillText(opts.value,L+tw+8,20);}
  };
  const pts=series.map(s=>s.xs.map((x,i)=>[x,opts.log?(s.ys[i]>0?Math.log10(s.ys[i]):NaN):s.ys[i]]).filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1])));
  const all=[].concat(...pts);
  if(!all.length){size(32);title(12);ctx.font='12px '+font;ctx.fillStyle=mut;ctx.textAlign='center';ctx.fillText('No data yet',w/2,32+ph/2);return;}
  const lo=(a,k)=>a.reduce((m,p)=>Math.min(m,p[k]),Infinity),hi=(a,k)=>a.reduce((m,p)=>Math.max(m,p[k]),-Infinity);
  let xmin=lo(all,0),xmax=hi(all,0);if(xmax===xmin)xmax=xmin+1;
  /* linear y: ignore the start-up transient (first 10% of iterations) so it doesn't flatten the converged part */
  const cut=xmin+0.1*(xmax-xmin),fit=!opts.log&&!opts.fitAll&&all.length>20?all.filter(p=>p[0]>=cut):all;
  let ymin=lo(fit,1),ymax=hi(fit,1),yt,ylab;
  if(opts.log){
    ymin=Math.floor(ymin);ymax=Math.ceil(ymax);if(ymax===ymin)ymax++;
    const st=Math.ceil((ymax-ymin)/6);yt=[];for(let e=ymin;e<=ymax;e+=st)yt.push(e);
    ylab=yt.map(fmtDecade);
  }else{
    const pad=(ymax-ymin)*0.08||Math.abs(ymax)*0.05||1;ymin-=pad;ymax+=pad;
    yt=niceTicks(ymin,ymax,4).filter(v=>v>=ymin&&v<=ymax);
    ylab=yt.map(v=>fmtNum(v,yt.length>1?yt[1]-yt[0]:Math.abs(v)||1));
  }
  /* left margin from the widest y label, so labels never clip */
  ctx.font='11px '+font;
  const L=Math.max(36,Math.ceil(Math.max(...ylab.map(s=>ctx.measureText(s).width)))+14),pw=w-L-R;
  const xt=niceTicks(xmin,xmax,Math.max(2,Math.floor(pw/90))).filter(v=>v>=xmin&&v<=xmax);
  const xstep=xt.length>1?xt[1]-xt[0]:1;
  /* legend: beside the title when it fits, else flowed onto rows of its own below it */
  ctx.font='600 12px '+font;let titleW=ctx.measureText(opts.title).width;
  ctx.font='12px '+font;if(opts.value)titleW+=8+ctx.measureText(opts.value).width;
  ctx.font='11px '+font;
  const leg=series.length>1?series.map((s,k)=>({k,name:s.name,w:16+ctx.measureText(s.name).width})):[];
  const legW=leg.reduce((a,l)=>a+l.w+12,0)-12,rows=[];
  if(leg.length&&titleW+24+legW>pw){
    let x=0,row=[];
    for(const l of leg){if(row.length&&x+l.w>pw){rows.push(row);row=[];x=0;}row.push(l);x+=l.w+12;}
    rows.push(row);
  }
  const T=32+rows.length*16;
  size(T);
  const px=x=>L+pw*(x-xmin)/(xmax-xmin),py=y=>T+ph-ph*(y-ymin)/(ymax-ymin);
  title(L);
  /* legend: right-aligned on the title row, or the wrapped rows under it */
  const swatch=(l,x,y)=>{ctx.fillStyle=col(l.k);ctx.fillRect(x,y-1,11,2);ctx.fillStyle=mut;ctx.fillText(l.name,x+16,y);};
  ctx.font='11px '+font;ctx.textAlign='left';ctx.textBaseline='middle';
  if(leg.length&&!rows.length){let x=L+pw-legW;for(const l of leg){swatch(l,x,16);x+=l.w+12;}}
  rows.forEach((row,r)=>{let x=L;for(const l of row){swatch(l,x,38+r*16);x+=l.w+12;}});
  /* grid and tick labels */
  ctx.lineWidth=1;ctx.strokeStyle=css('--bd');ctx.fillStyle=mut;
  ctx.textAlign='right';ctx.textBaseline='middle';
  yt.forEach((v,i)=>{const Y=Math.round(py(v))+0.5;ctx.beginPath();ctx.moveTo(L,Y);ctx.lineTo(L+pw,Y);ctx.stroke();ctx.fillText(ylab[i],L-8,Y);});
  ctx.textAlign='center';ctx.textBaseline='top';
  for(const v of xt){const X=Math.round(px(v))+0.5;ctx.beginPath();ctx.moveTo(X,T);ctx.lineTo(X,T+ph);ctx.stroke();ctx.fillText(fmtNum(v,xstep),X,T+ph+6);}
  ctx.fillText(opts.xlabel||'Iteration',L+pw/2,T+ph+20);
  ctx.strokeStyle=css('--bd-hi');ctx.beginPath();ctx.moveTo(L,T+ph+0.5);ctx.lineTo(L+pw,T+ph+0.5);ctx.stroke();
  /* series, clipped to the plot area; a dot marks each series' latest point */
  ctx.save();ctx.beginPath();ctx.rect(L,T,pw,ph);ctx.clip();ctx.lineWidth=1.75;ctx.lineJoin=ctx.lineCap='round';
  const dot=(p,r)=>{ctx.beginPath();ctx.arc(px(p[0]),py(p[1]),r,0,2*Math.PI);ctx.fill();};
  series.forEach((s,k)=>{
    if(!pts[k].length)return;
    ctx.strokeStyle=ctx.fillStyle=col(k);ctx.beginPath();
    pts[k].forEach((p,i)=>i?ctx.lineTo(px(p[0]),py(p[1])):ctx.moveTo(px(p[0]),py(p[1])));ctx.stroke();
    if(opts.dots)pts[k].forEach(p=>dot(p,3));else dot(pts[k][pts[k].length-1],3);
  });
  ctx.restore();
  /* hover: crosshair at the point nearest the cursor, and a readout box of every series there */
  const hx=canvas._hx;
  if(hx==null||hx<L||hx>L+pw)return;
  const xv=xmin+(hx-L)/pw*(xmax-xmin),near=pts.map(a=>a.reduce((b,p)=>!b||Math.abs(p[0]-xv)<Math.abs(b[0]-xv)?p:b,null));
  const ref=near.find(Boolean);if(!ref)return;
  const X=Math.round(px(ref[0]))+0.5;
  ctx.strokeStyle=css('--bd-hi');ctx.beginPath();ctx.moveTo(X,T);ctx.lineTo(X,T+ph);ctx.stroke();
  const bg=css('--bg-elev');
  near.forEach((p,k)=>{if(!p)return;ctx.fillStyle=col(k);dot(p,4);ctx.strokeStyle=bg;ctx.lineWidth=1.5;ctx.stroke();});
  const lines=[[null,(opts.xlabel||'Iteration').replace(/&[^;]+;/g,'')+' '+(Number.isInteger(ref[0])?ref[0].toLocaleString():fmtNum(ref[0]))]].concat(
    near.map((p,k)=>p&&[k,series[k].name+'  '+fmtNum(opts.log?Math.pow(10,p[1]):p[1])]).filter(Boolean));
  ctx.font='11px '+font;
  const bw=Math.max(...lines.map(([k,t])=>ctx.measureText(t).width+(k==null?0:16)))+16,bh=lines.length*16+8;
  let bx=X+10;if(bx+bw>w-2)bx=X-10-bw;bx=Math.max(2,bx);
  const by=T+4;
  ctx.fillStyle=bg;ctx.strokeStyle=css('--bd-hi');ctx.lineWidth=1;
  ctx.beginPath();ctx.roundRect?ctx.roundRect(bx,by,bw,bh,6):ctx.rect(bx,by,bw,bh);ctx.fill();ctx.stroke();
  ctx.textAlign='left';ctx.textBaseline='middle';
  lines.forEach(([k,t],i)=>{
    const y=by+12+i*16;
    if(k==null){ctx.fillStyle=mut;ctx.fillText(t,bx+8,y);}
    else{ctx.fillStyle=col(k);ctx.fillRect(bx+8,y-1,11,2);ctx.fillStyle=fg;ctx.fillText(t,bx+24,y);}
  });
}
/* a .seg whose buttons pick what a chart plots: opts [[key,label,tip]], onPick(key) */
function segPicker(el,opts,onPick){
  el.innerHTML=opts.map(([k,t,tip],i)=>'<button data-k="'+k+'"'+(i?'':' class="on"')+' data-tip="'+tip+'">'+t+'</button>').join('');
  el.querySelectorAll('button').forEach(b=>b.onclick=()=>{
    el.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));onPick(b.dataset.k);
  });
  ariaTips();
}
const COEF={Cd:'Drag coefficient Cd',Cl:'Lift coefficient Cl',Cm:'Pitching moment coefficient Cm (nose-up +)'};
let CHART_COEF='Cd';
segPicker($('coefseg'),[['Cd','Cd','Plot the drag coefficient.'],['Cl','Cl','Plot the lift coefficient.'],
  ['Cm','Cm','Plot the pitching moment coefficient about the moment reference point, positive nose-up.']],
  k=>{CHART_COEF=k;if(LAST_HIST)drawChart(LAST_HIST);});
const xLabel=h=>h&&h.transient?'Time (s)':'Iteration';
function drawChart(hist){
  const series=h=>{const cols=h.coeffs.columns,rows=h.coeffs.rows,ci=cols.indexOf(CHART_COEF);
    return {xs:rows.map(r=>r[0]),ys:ci<0?[]:rows.map(r=>r[ci])};};
  const a=series(hist),last=a.ys[a.ys.length-1],cmp=featOn('compare')&&CMP;   /* Compare off: no overlay */
  const s=[{name:cmp?caseLabel(CUR):CHART_COEF,...a,color:getComputedStyle(document.body).getPropertyValue('--ac').trim()}];
  /* the compared case, when its x axis means the same (iterations vs simulated seconds) */
  if(cmp&&CMP.hist&&!!CMP.hist.transient===!!hist.transient)s.push({name:caseLabel(CMP.name),...series(CMP.hist),color:SERIES_COL[1]});
  plot($('chart'),s,{title:COEF[CHART_COEF],value:last!=null?last.toFixed(4):'',xlabel:xLabel(hist)});
}
/* initial residuals per field, log10 scale */
function drawResiduals(res){
  const rows=res?res.rows:[],cols=res?res.columns:[];
  plot($('reschart'),cols.slice(1).map((name,k)=>({name,xs:rows.map(r=>r[0]),ys:rows.map(r=>r[k+1])})),
    {title:'Initial residuals',log:true,xlabel:xLabel(LAST_HIST)});
}
/* probes: point histories and sample-line profiles of the quantity picked in #probeqseg */
const PQ=[['|U|','|U|','Velocity magnitude (m/s).'],['Ux','Ux','Velocity component along x (m/s).'],['Uy','Uy','Velocity component along y (m/s).'],
  ['Uz','Uz','Velocity component along z (m/s).'],['p','p','Kinematic pressure p (m²/s², as the solver works with it; 0 at the outlet).']];
let PROBE_Q='|U|';
segPicker($('probeqseg'),PQ,k=>{PROBE_Q=k;if(LAST_HIST)drawProbes(LAST_HIST);});
function drawProbes(h){
  const pr=h.probes&&h.probes.t.length?h.probes:null,ln=h.lines&&h.lines.lines.length?h.lines:null;
  const unit=PROBE_Q==='p'?' (m²/s²)':' (m/s)';
  $('linechart').style.display=ln?'':'none';$('probechart').style.display=pr?'':'none';
  if(ln)plot($('linechart'),ln.lines.map(l=>({name:l.name.replace('line','Line '),xs:l.d,ys:l.q[PROBE_Q]})),
    {title:PROBE_Q+unit+' along the sample lines at '+(h.transient?ln.time+' s':'iteration '+ln.time),xlabel:'Distance from start (m)',fitAll:true});
  if(pr)plot($('probechart'),pr.names.map((n,i)=>({name:n,xs:pr.t,ys:pr.q[PROBE_Q][i]})),
    {title:PROBE_Q+unit+' at the probe points',xlabel:xLabel(h)});
  $('probenote').textContent=pr||ln?'':'No probe data. Set probe points or sample lines under Run → Probes, then solve (or Sample lines).';
  $('probenote').style.display=pr||ln?'none':'';
}
let LAST_HIST=null;   /* last history drawn, so the charts can be redrawn sharp at a new width */
window.addEventListener('resize',()=>document.querySelectorAll('.chartcanvas').forEach(drawPlot));
/* the embedded fonts load after the first paint: redraw canvas text and re-measure the header once they're in */
if(document.fonts)document.fonts.ready.then(()=>dispatchEvent(new Event('resize')));
function renderTable(hist){
  const cols=hist.coeffs.columns,rows=hist.coeffs.rows.slice(-50),dimRows=hist.dimensional_coeffs.rows.slice(-50);
  if(!rows.length){$('coefftable').innerHTML='';}
  else{
    let html='<table><thead><tr><th>Time</th>'+cols.slice(1).map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
    rows.forEach((row,i)=>{
      html+='<tr><td>'+row[0].toPrecision(4)+'</td>'+row.slice(1).map((v,j)=>{
        const d=dimRows[i]?dimRows[i][j+1]:null;
        const unit=cols[j+1].startsWith('Cm')?'Nm':'N';
        return '<td>'+v.toFixed(4)+(d!=null?' <span class="note">('+d.toFixed(2)+' '+unit+')</span>':'')+'</td>';
      }).join('')+'</tr>';
    });
    $('coefftable').innerHTML=html+'</tbody></table>';
  }
  const frows=hist.forces.rows;
  $('forceline').textContent=frows.length?('Latest forces @ t='+frows[frows.length-1][0].toPrecision(4)+': Fx='+frows[frows.length-1][1].toFixed(3)+' N, Fy='+frows[frows.length-1][2].toFixed(3)+' N, Fz='+frows[frows.length-1][3].toFixed(3)+' N, Mx='+frows[frows.length-1][4].toFixed(3)+' Nm, My='+frows[frows.length-1][5].toFixed(3)+' Nm, Mz='+frows[frows.length-1][6].toFixed(3)+' Nm'):'No force.dat data yet.';
}
/* averaged coefficients and forces (mean over the last N iterations, with the min..max range) as stat chips */
function renderAverages(av,flight){
  const el=$('avgstats'),rows=[],heads=[];
  if(av){
    /* Cd, Cl and Cm are the headline readouts; any other coefficient is a chip */
    for(const [k,a] of Object.entries(av.coeffs)){
      const d=av.dimensional[k],spread='&plusmn;'+((a.max-a.min)/2).toPrecision(2)+(d?' · '+d.mean.toPrecision(4)+(k.startsWith('Cm')?' N m':' N'):'');
      if(['Cd','Cl','Cm'].includes(k))heads.push('<div class="readout"><b>'+esc(k)+'</b><span class="num">'+a.mean.toFixed(4)+'</span><small>'+spread+'</small></div>');
      else rows.push([k,a.mean.toFixed(4)+' <span class="note">'+spread+'</span>']);
    }
    for(const k of ['Fx','Fy','Fz'])if(av.forces[k])rows.push([k,av.forces[k].mean.toPrecision(4)+' N']);
    if(av.window_s!=null)rows.push(['Averaged over','last '+(+av.window_s.toPrecision(3))+' s']);
  }
  /* transient: the shedding frequency from the lift and side-force oscillations */
  const u=HIST&&HIST.unsteady;
  if(av&&u){
    const st=(s,what)=>s?(+s.St.toPrecision(3))+' <span class="note">'+(+s.f.toPrecision(3))+' Hz · '+s.cycles+' cycles</span>':'none ('+what+' not oscillating yet)';
    rows.push(['Strouhal (lift)',st(u.lift,'Cl')]);
    if('side' in u)rows.push(['Strouhal (side F'+u.side_axis+')',st(u.side,'side force')]);
  }
  /* with an aircraft mass: wing loading and the speed at which this Cl holds the aircraft up */
  if(av&&flight){
    rows.push(['Wing loading',flight.wing_loading.toPrecision(3)+' N/m&sup2;']);
    rows.push(['Level-flight speed',flight.v_level?flight.v_level.toPrecision(3)+' m/s':'none (Cl &le; 0)']);
  }
  el.innerHTML=(heads.length?'<div class="readouts">'+heads.join('')+'</div>':'')+
    (rows.length?'<div class="stats">'+rows.map(([k,v])=>'<div class="stat"><b>'+esc(k)+'</b>'+v+'</div>').join('')+'</div>':'');
}
const avgWindow=()=>Math.max(1,+$('avg_window').value||100);
/* the averaging window as query parameters: iterations (steady) and seconds (transient; blank = the case's default) */
const avgQuery=()=>'&avg='+avgWindow()+(+$('avg_time').value>0?'&avg_time='+(+$('avg_time').value):'');
['avg_window','avg_time'].forEach(id=>$(id).addEventListener('change',()=>refreshCaseView()));
function exportCase(kind){
  if(!CUR)return;
  const url='/api/'+kind+'?case='+encodeURIComponent(CUR)+avgQuery();
  if(kind==='report')open(url,'_blank');else location.href=url;
}
async function refreshCaseView(){
  const c=CUR;if(!c)return;
  const q='?case='+encodeURIComponent(c)+avgQuery();
  const [ms,hist,res]=await Promise.all([j('/api/mesh_status'+q),j('/api/history'+q+'&mass='+massKg()),j('/api/results'+q)]);
  if(c!==CUR)return;   /* the case was switched while this was loading */
  renderMeshStatus(ms.ok?ms:null);
  /* an uncreated case has no history: render the empty state so nothing stale shows */
  const h=hist.ok?hist:{coeffs:{columns:[],rows:[]},dimensional_coeffs:{rows:[]},forces:{rows:[]},residuals:null,averages:null};
  HIST=h;LAST_HIST=h;
  if(CMP&&CMP.name!==$('cmp_case').value)CMP=null;
  if($('cmp_case').value&&(!CMP||JOBSTATE&&[c,$('cmp_case').value].includes(JOBSTATE.case)))await loadCompare();
  if(c!==CUR)return;
  drawChart(h);drawResiduals(h.residuals);renderTable(h);renderAverages(h.averages,h.flight);renderPhysics();drawProbes(h);renderCompare();
  $('results').textContent=res.text||'';
  /* Results unlock as soon as the solver has written coefficients (also live, mid-solve) */
  $('rescard').classList.toggle('locked',!h.coeffs.rows.length);
  renderSteps();
}
/* ---------- case comparison: another solved case's averages, settings and history beside this one ---------- */
let CMP=null;   /* {name, hist} of the compared case */
/* the cases offered by a case picker: solved ones (or all), the current case's folder first */
function casePicker(el,first,keep,filter){
  const cur=keep!=null?keep:el.value,p=folderOf(CUR||'');
  const list=CASES.filter(c=>c.name!==CUR&&filter(c));
  const grp=(t,l)=>l.length?'<optgroup label="'+esc(t)+'">'+l.map(c=>'<option value="'+esc(c.name)+'">'+esc(caseLabel(c.name))+'</option>').join('')+'</optgroup>':'';
  el.innerHTML=first+grp(p?'This folder: '+caseLabel(p):'Top level',list.filter(c=>c.folder===p))+grp('Other cases',list.filter(c=>c.folder!==p));
  if(cur&&!Array.from(el.options).some(o=>o.value===cur))   /* keep a saved choice that isn't listed (not solved yet) */
    el.insertAdjacentHTML('beforeend','<option value="'+esc(cur)+'">'+esc(caseLabel(cur))+' (no results)</option>');
  el.value=cur||'';
}
function casePickers(){
  casePicker($('cmp_case'),'<option value="">None</option>',null,c=>c.has_results);
  casePicker($('init_case'),'<option value="">Uniform free stream</option>',null,c=>c.has_results);
  initNote();
}
/* a warm start from a case with another turbulence model only carries the fields both models share */
function initNote(){
  const c=CASES.find(x=>x.name===$('init_case').value),tm=$('turb_model').value;
  setAlert($('initnote'),'warn',c&&c.turb_model&&c.turb_model!==tm?esc(caseLabel(c.name))+' was solved with '+esc(TURB_MODELS[c.turb_model]||c.turb_model)+
    ', this case uses '+esc(TURB_MODELS[tm])+': only the fields both share (U, p, ν<sub>t</sub>) carry over, the rest start uniform.':'');
}
['init_case','turb_model'].forEach(id=>$(id).addEventListener('change',initNote));
async function loadCompare(){
  const name=$('cmp_case').value;
  if(!name){CMP=null;return;}
  const r=await j('/api/history?case='+encodeURIComponent(name)+avgQuery());
  CMP={name,hist:r.ok?r:null,error:r.ok?null:r.error};
}
$('cmp_case').addEventListener('change',async()=>{await loadCompare();if(LAST_HIST){drawChart(LAST_HIST);renderCompare();}});
const PARAM_LABEL=k=>{const el=$(k),w=el&&el.closest('[data-tip]'),l=w&&w.querySelector('label');return l?l.textContent.trim():k;};
function renderCompare(){
  const show=(id,v)=>{$(id).style.display=v?'':'none';};
  if(!CMP){$('cmpnote').textContent='Pick a case under Compare with to see it next to this one.';show('cmpnote',1);$('cmptable').innerHTML='';show('cmpdiffbox',0);return;}
  if(!CMP.hist){$('cmpnote').textContent=CMP.error||'No results for that case.';show('cmpnote',1);$('cmptable').innerHTML='';show('cmpdiffbox',0);return;}
  const a=LAST_HIST.averages,b=CMP.hist.averages,f=v=>v==null?'–':Math.abs(v)>=100?v.toPrecision(5):v.toFixed(4);
  const rows=[];
  for(const grp of ['coeffs','forces'])for(const k of new Set([...Object.keys(a?a[grp]:{}),...Object.keys(b?b[grp]:{})])){
    const x=a&&a[grp][k]?a[grp][k].mean:null,y=b&&b[grp][k]?b[grp][k].mean:null;
    rows.push([k+(grp==='forces'?' (N'+(k[0]==='M'?' m':'')+')':''),f(x),f(y),x!=null&&y!=null&&x?((y-x)/Math.abs(x)*100).toFixed(1)+' %':'–']);
  }
  const win=h=>h.averages?(h.averages.window_s!=null?'last '+(+h.averages.window_s.toPrecision(3))+' s':'last '+h.averages.window+' iterations'):'';
  $('cmpnote').textContent='Means over each case\'s '+(LAST_HIST.transient||CMP.hist.transient?'averaging window':'last iterations')+': '+
    caseLabel(CUR)+' '+win(LAST_HIST)+', '+caseLabel(CMP.name)+' '+win(CMP.hist)+'. Δ is relative to this case.'+
    (!!CMP.hist.transient!==!!LAST_HIST.transient?' One case is transient and one steady, so their histories aren\'t overlaid on the convergence chart.':'');show('cmpnote',1);
  $('cmptable').innerHTML='<table><thead><tr><th>Quantity</th><th>'+esc(caseLabel(CUR))+'</th><th>'+esc(caseLabel(CMP.name))+'</th><th>Δ</th></tr></thead><tbody>'+
    rows.map(r=>'<tr>'+r.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  /* settings: the written params of both cases as they were used (switched-off features neutral), keys that differ */
  const pa=effParams((LAST_HIST.info||{}).params||{}),pb=effParams((CMP.hist.info||{}).params||{});
  const diff=PARAMS.filter(k=>k!=='case'&&k!=='features'&&String(pa[k]??'')!==String(pb[k]??''));
  show('cmpdiffbox',1);
  $('cmpdiff').innerHTML=diff.length?'<table><thead><tr><th>Setting</th><th>'+esc(caseLabel(CUR))+'</th><th>'+esc(caseLabel(CMP.name))+'</th></tr></thead><tbody>'+
    diff.map(k=>'<tr><td>'+esc(PARAM_LABEL(k))+'</td><td>'+esc(pa[k]??'–')+'</td><td>'+esc(pb[k]??'–')+'</td></tr>').join('')+'</tbody></table>':
    '<div class="note">Both cases were created with the same settings.</div>';
}

/* ---------- parameter sweep ---------- */
async function runSweep(){
  const param=$('sweep_param').value;
  const values=$('sweep_values').value.split(',').map(s=>s.trim()).filter(Boolean);
  if(!values.length){alert('Enter at least one value.');return;}
  const r=await j('/api/run',{method:'POST',body:JSON.stringify({action:'sweep',param,values,base:params(),mesh:meshCfg(),
    autostop:autostopCfg(),nprocs:+$('nprocs').value,avg:avgWindow()})});
  if(!r.ok)alert(r.error);
}
const SWEEP_KEY=new Set(['variant','status','cells','Time']);
function renderSweep(s){
  SWEEP=s;renderSteps();renderAero(s);
  const chart=$('sweepchart');
  if(!(s&&(s.polar||s.yaw))){$('sweepseg').innerHTML='';delete $('sweepseg').dataset.keys;}
  if(!s){$('sweepprogress').textContent='';$('sweeptable').innerHTML='';chart.style.display='none';return;}
  $('sweepprogress').textContent=s.running?('Running variant '+s.sweep_index+'/'+s.sweep_total+' ('+s.sweep_value+')'):
    (s.results?('Done - '+s.results.length+' variant(s), '+esc(s.param)+'; values are means over the last '+s.avg+' iterations'):'');
  const rows=s.results||[];
  if(!rows.length){$('sweeptable').innerHTML='';chart.style.display='none';return;}
  /* ΔCd: change of the mean Cd from the previous variant - the mesh-independence measure */
  rows.forEach((r,i)=>{const p=rows[i-1];r['ΔCd %']=p&&p.Cd!=null&&r.Cd!=null&&p.Cd?100*(r.Cd-p.Cd)/Math.abs(p.Cd):null;});
  const keys=['variant','status','cells','Time'].concat([...new Set(rows.flatMap(Object.keys))].filter(k=>!SWEEP_KEY.has(k)&&k!=='case'&&k!=='error'));
  const cell=(k,v)=>v==null?'':typeof v!=='number'?esc(v):k==='cells'?v.toLocaleString():k==='Time'?String(v):k==='ΔCd %'?v.toFixed(2):v.toFixed(4);
  $('sweeptable').innerHTML='<table><thead><tr>'+keys.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr></thead><tbody>'+
    rows.map(r=>'<tr>'+keys.map(k=>'<td'+(r.error&&k==='status'?' data-tip="'+esc(r.error)+'"':'')+'>'+cell(k,r[k])+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  if(s.polar||s.yaw){drawSweepChart();return;}
  /* Cd vs cell count for a mesh study, vs the swept value when it's numeric */
  const mesh=['mesh_preset','cell_size_mm','target_cells','surface_cell_mm'].includes(s.param),ok=rows.filter(r=>r.Cd!=null&&(mesh?r.cells:typeof r.variant==='number'));
  chart.style.display=ok.length?'':'none';
  if(ok.length)plot(chart,[{name:'Cd',xs:ok.map(r=>mesh?r.cells:r.variant),ys:ok.map(r=>r.Cd)}],
    {title:'Mean Cd per variant',xlabel:mesh?'Cells':esc(s.param),dots:true});
}
/* aircraft sweeps: the polar / yaw-stability charts, picked with a .seg ([key, label, tip, x, y, title, xlabel]) */
const POLAR_VIEWS=[['cla','CL–α','Lift coefficient against angle of attack: the slope is the lift curve slope, the peak is stall.','alpha','CL','Lift coefficient CL','Angle of attack (°)'],
  ['cda','CD–α','Drag coefficient against angle of attack.','alpha','CD','Drag coefficient CD','Angle of attack (°)'],
  ['cma','Cm–α','Pitching moment about the CG against angle of attack. A downward slope means the aircraft is statically stable in pitch.','alpha','Cm','Pitching moment coefficient Cm (nose-up +)','Angle of attack (°)'],
  ['polar','Drag polar','Lift against drag coefficient. The tangent from the origin touches it at the best lift-to-drag ratio.','CD','CL','Drag polar: CL against CD','Drag coefficient CD'],
  ['ld','L/D–α','Lift-to-drag ratio against angle of attack: its peak is the best glide angle.','alpha','LD','Lift-to-drag ratio L/D','Angle of attack (°)']];
const YAW_VIEWS=[['cnb','Cn–β','Yawing moment against sideslip. A rising line (Cnβ > 0) means the nose turns into the wind: directionally stable.','beta','Cn','Yawing moment coefficient Cn','Sideslip β (°)'],
  ['clb','Cl–β','Rolling moment against sideslip. A falling line (Clβ < 0) means the aircraft rolls away from the sideslip: a stable dihedral effect.','beta','Croll','Rolling moment coefficient Cl','Sideslip β (°)']];
let SWEEP_VIEW=null;
function drawSweepChart(){
  const s=SWEEP,d=s&&(s.polar||s.yaw),views=s&&s.polar?POLAR_VIEWS:YAW_VIEWS,seg=$('sweepseg'),chart=$('sweepchart');
  if(!d){chart.style.display='none';return;}
  const keys=views.map(v=>v[0]).join();
  if(seg.dataset.keys!==keys){seg.dataset.keys=keys;SWEEP_VIEW=views[0][0];
    segPicker(seg,views.map(v=>[v[0],v[1],v[2]]),k=>{SWEEP_VIEW=k;drawSweepChart();});}
  const v=views.find(x=>x[0]===SWEEP_VIEW)||views[0],pts=d.points.filter(p=>p[v[3]]!=null&&p[v[4]]!=null);
  chart.style.display='';
  plot(chart,[{name:v[4],xs:pts.map(p=>p[v[3]]),ys:pts.map(p=>p[v[4]])}],{title:v[5],xlabel:v[6],dots:true});
}
/* derived aircraft figures as chips; verdicts are spelled out, never colour alone */
function renderAero(s){
  const el=$('aerostats'),note=$('aeronote'),c=[],f=(v,n)=>v==null?null:(+v.toPrecision(n||3)).toString();
  const add=(k,v)=>{if(v!=null)c.push([k,v]);};
  const p=s&&s.polar,y=s&&s.yaw;let txt='';
  if(p){
    add('CLα',f(p.cl_alpha)&&f(p.cl_alpha)+' /°');add('Zero-lift α',f(p.alpha0)&&f(p.alpha0)+'°');
    add('CLmax',f(p.cl_max)+' at '+f(p.alpha_stall)+'°'+(p.stall_reached?'':' (stall not reached)'));
    add('Best L/D',p.ld_max!=null?f(p.ld_max)+' fitted':f(p.ld_max_pts)+' at '+f(p.alpha_ld_max)+'°');
    add('CD0',f(p.cd0));add('Oswald e',f(p.oswald,2));
    if(p.static_margin!=null){
      add('Static margin',f(p.static_margin*100)+' % '+(p.static_margin>0?'(stable)':'(unstable)'));
      add('Neutral point',p.flow_axis.toUpperCase()+' = '+f(p.neutral_point,4)+' m');
      add('CG for '+f(p.target_sm)+' %',p.flow_axis.toUpperCase()+' = '+f(p.cg_target,4)+' m');
    }
    add('Cmα',f(p.cm_alpha)&&f(p.cm_alpha)+' /°');
    add('Trim',p.alpha_trim!=null?f(p.alpha_trim)+'° (CL '+f(p.cl_trim)+')':'no Cm = 0 in range');
    const q=p.perf;
    if(q){
      add('Wing loading',f(q.wing_loading)+' N/m²');add('Stall speed',f(q.v_stall)+' m/s'+(p.stall_reached?'':' or less'));
      add('Trimmed speed',q.v_trim&&f(q.v_trim)+' m/s');
      add('Best glide',q.v_best_glide&&f(q.v_best_glide)+' m/s, 1 : '+f(q.glide_ratio));
      add('Min sink',q.v_min_sink&&f(q.v_min_sink)+' m/s, '+f(q.sink_rate)+' m/s down');
    }
    txt='Figures are fitted to the variants; the CG positions are along the flow axis in the part\'s coordinates. '+
      (q?'Speeds use the sweep\'s Reynolds number and assume the aircraft is re-trimmed at each speed.':'Enter the aircraft mass under Physics for flight speeds.');
  }else if(y){
    const v=(k,good)=>y[k]==null?null:f(y[k])+' /rad '+(good(y[k])?'(stable)':'(unstable)');
    add('Cnβ',v('cn_beta',x=>x>0));add('Clβ',v('cl_beta',x=>x<0));add('CYβ',y.cy_beta!=null?f(y.cy_beta)+' /rad':null);
    txt='Body axes: x forward, y right wing, z down; β positive with the wind from the right. Moments are about the moment reference point.';
  }
  el.className=c.length?'stats':'';
  el.innerHTML=c.map(([k,v])=>'<div class="stat"><b>'+esc(k)+'</b>'+esc(v)+'</div>').join('');
  note.textContent=txt;
}
async function refreshSweep(){
  renderSweep(await j('/api/sweep_status?mass='+massKg()+'&target_sm='+(+fv('target_sm')||10)));
}

/* ---------- completion notifications (frontend-only, anti-duplicate) ---------- */
let notifyOn=false;
function enableNotify(){
  if(!('Notification' in window)){alert('Notifications not supported in this browser.');return;}
  Notification.requestPermission().then(p=>{notifyOn=p==='granted';$('notifybtn').textContent=notifyOn?'Notifications on':'Enable notifications';});
}
function beep(){
  try{
    const Ctx=window.AudioContext||window.webkitAudioContext,actx=new Ctx();
    const o=actx.createOscillator(),g=actx.createGain();
    o.frequency.value=880;o.connect(g);g.connect(actx.destination);
    g.gain.setValueAtTime(0.15,actx.currentTime);
    o.start();o.stop(actx.currentTime+0.18);
  }catch(e){}
}
const origTitle=document.title;let flashTimer=null;
function flashTitle(msg){
  if(flashTimer)clearInterval(flashTimer);
  let on=false;flashTimer=setInterval(()=>{document.title=on?origTitle:msg;on=!on;},800);
  const stop=()=>{clearInterval(flashTimer);flashTimer=null;document.title=origTitle;window.removeEventListener('focus',stop);};
  window.addEventListener('focus',stop);setTimeout(stop,15000);
}
let lastSeq=null;
function checkCompletion(s){
  if(!s.last_finish)return;
  if(lastSeq===null){lastSeq=s.last_finish.seq;return;}
  if(s.last_finish.seq>lastSeq){
    lastSeq=s.last_finish.seq;
    const info=s.last_finish,msg=info.kind+' '+(info.ok?'finished':'stopped/failed')+' - '+info.case;
    if(notifyOn&&Notification.permission==='granted')new Notification('OpenFOAM GUI',{body:msg});
    beep();flashTitle(msg);
    if(info.case===CUR)refreshCaseView();
    loadCases();
  }
}

/* ---------- server restart ---------- */
let BOOT=null,QUEUED=0,RESTARTING=false;   /* server process id from /api/state; the poll reloads when it changes */
async function restartServer(){
  let msg='Restart the PIFOAM server?';
  if(JOBSTATE)msg+='\n\nThis stops '+JOBSTATE.kind+' on '+JOBSTATE.case+' first (a solver writes its current fields).';
  if(QUEUED)msg+='\n'+QUEUED+' queued job(s) will be dropped.';
  if(!confirm(msg))return;
  const r=await j('/api/restart',{method:'POST'});
  if(!r.ok)return alert(r.error);
  RESTARTING=true;$('statustext').textContent='Restarting…';
}

/* ---------- polling ---------- */
setInterval(async()=>{
  tick++;
  /* a slow server (a busy mesher) lets the next poll start before this one returns: only the response
     that continues from the current position may append, or the same log chunk shows up twice */
  const since=pos,s=await j('/api/state?since='+since);
  if(since===pos){
    if(s.log){const l=$('log');l.textContent+=s.log;l.scrollTop=l.scrollHeight;}
    pos=s.pos;
  }
  /* a new server process (after Restart) serves a new page: reload onto it */
  if(BOOT!==null&&s.boot!==BOOT){location.reload();return;}
  BOOT=s.boot;QUEUED=(s.queue||[]).length;
  JOBSTATE=s.running&&s.job?Object.assign({now:s.now},s.job):null;
  const pr=JOBSTATE&&progress(JOBSTATE.meta),step=JOBSTATE&&JOBSTATE.meta.step;
  $('statustext').textContent=RESTARTING?'Restarting…':!s.running?'Idle':pr?('Solving '+pr.sofar+(pr.left!=null?' · '+dur(pr.left)+' left':'')+' · '+s.job.case):
    ('Running '+(s.job?s.job.kind+' · '+s.job.case+(step?' · '+step:''):''));
  $('status').classList.toggle('run',!!s.running);
  /* solver progress bar, for the case on screen */
  const mine=pr&&JOBSTATE.case===CUR;
  $('progwrap').style.display=mine?'':'none';
  if(mine){$('solveprog').value=pr.frac;$('progtext').textContent=pr.sofar+' '+(pr.unit==='iterations'?'iterations ':'')+'('+Math.round(pr.frac*100)+'%)'+(pr.left!=null?' · about '+dur(pr.left)+' left':'');}
  renderSteps();
  renderQueue(s.queue);
  checkCompletion(s);
  if(s.running&&tick%3===0)refreshCaseView();
},1000);
setInterval(refreshSweep,2000);
syncFeatures();loadCases();
</script></body></html>"""


# ---------------------------------------------------------------- server
def json_safe(obj):
    """obj with NaN/inf floats (a diverged solve's results) as None: json.dumps writes them as bare NaN,
    which the browser's JSON parser rejects, failing the whole response."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, ctype="application/json", code=200, headers=None):
        body = obj if isinstance(obj, bytes) else json.dumps(json_safe(obj)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            page = (PAGE.replace("/*FONTS*/", FONTS)
                        .replace("/*MATERIALS*/null", json.dumps(MATERIALS))
                        .replace("/*SIM_TYPES*/null", json.dumps(SIM_TYPES))
                        .replace("/*TURB_MODELS*/null", json.dumps(TURB_MODELS))
                        .replace("/*FEATURE_GROUPS*/null", json.dumps(FEATURE_GROUPS))
                        .replace("/*NPROCS*/1", str(DEFAULT_NPROCS))
                        .replace("/*CORES*/[1,1]", json.dumps([CORES, THREADS])))
            return self.send(page.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            since = int(q.get("since", ["0"])[0])
            text = "".join(LOG)
            if since > len(text):
                since = 0
            return self.send({
                "log": text[since:], "pos": len(text), "running": JOB["running"],
                "job": {"kind": JOB["kind"], "case": JOB["case"], "meta": JOB["meta"]} if JOB["running"] else None,
                "last_finish": JOB["last_finish"], "queue": queue_snapshot(), "now": time.time(), "boot": BOOT,
            })
        try:   # iterations the coefficients are averaged over
            avg = max(1, int(q.get("avg", [AVG_WINDOW])[0]))
        except ValueError:
            avg = AVG_WINDOW
        try:   # seconds a transient case is averaged over (blank: the case's default)
            avg_time = float(q.get("avg_time", [""])[0] or 0) or None
        except ValueError:
            avg_time = None
        if u.path == "/api/results":
            try:
                return self.send(results(case_path(q.get("case", [""])[0]), avg, avg_time))
            except Exception as e:
                return self.send({"error": str(e)})
        if u.path == "/api/history":
            try:
                mass = q.get("mass", [""])[0]
                return self.send({"ok": True, **history(case_path(q.get("case", [""])[0]), avg,
                                                        float(mass) if mass else None, avg_time)})
            except Exception as e:
                return self.send({"error": str(e)})
        if u.path in ("/api/export.csv", "/api/report"):
            try:
                case = case_path(q.get("case", [""])[0])
            except ValueError as e:
                return self.send(str(e).encode(), "text/plain; charset=utf-8", 404)
            name = os.path.basename(case)
            if u.path == "/api/report":
                return self.send(report_html(case, avg, avg_time).encode(), "text/html; charset=utf-8")
            return self.send(history_csv(case).encode(), "text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{name}_history.csv"'})
        if u.path == "/api/mesh_status":
            try:
                return self.send({"ok": True, **read_mesh_status(case_path(q.get("case", [""])[0]))})
            except Exception as e:
                return self.send({"error": str(e)})
        if u.path == "/api/cases":
            return self.send({"ok": True, "cases": list_cases(), "folders": list_folders()})
        if u.path == "/api/thumb":   # ?case= a case's part, or ?folder= a folder's shared part
            try:
                if "folder" in q:
                    stl = folder_stl(folder_dir(q["folder"][0])[0])[0] or ""
                else:
                    stl = os.path.join(case_path(q.get("case", [""])[0]), "constant/geometry/part.stl")
                return self.send({"ok": True, "preview": stl_thumb(stl)})
            except (OSError, ValueError) as e:
                return self.send({"ok": False, "error": str(e)})
        if u.path == "/api/sweep_status":
            if JOB["running"] and JOB["kind"] == "sweep":
                return self.send({"ok": True, "running": True, **JOB["meta"]})
            out = {"ok": True, "running": False, **LAST_SWEEP}
            rows, ref = LAST_SWEEP.get("results") or [], LAST_SWEEP.get("ref")
            try:   # aircraft figures: computed here, so a new mass or target margin needs no re-run
                mass = float(q.get("mass", ["0"])[0] or 0)
                target_sm = float(q.get("target_sm", ["10"])[0] or 10)
            except ValueError:
                mass, target_sm = 0, 10
            if LAST_SWEEP.get("param") == "aoa":
                out["polar"] = polar_summary(rows, ref, mass, target_sm)
            elif LAST_SWEEP.get("param") == "yaw":
                out["yaw"] = yaw_summary(rows, ref)
            return self.send(out)
        self.send({"error": "not found"}, code=404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/api/upload":
                return self.send({"ok": True, **load_stl(self.body(), q.get("name", [""])[0])})
            if u.path == "/api/folders/stl":
                set_folder_stl(q.get("folder", [""])[0], self.body(), q.get("name", [""])[0],
                               q.get("scale", ["0.001"])[0])
                return self.send({"ok": True})
            if u.path.startswith("/api/folders/"):
                req = json.loads(self.body() or b"{}")
                act = u.path.rsplit("/", 1)[1]
                if act == "create":
                    return self.send({"ok": True, "name": create_folder(req.get("name"), req.get("parent") or "")})
                if act == "rename":
                    return self.send({"ok": True, "name": rename_folder(req.get("name"), req.get("new"))})
                if act == "move":
                    return self.send({"ok": True, "name": move_folder(req.get("name"), req.get("dest") or "")})
                if act == "delete":
                    delete_folder(req.get("name"), bool(req.get("with_cases")))
                    return self.send({"ok": True})
                if act == "use_stl":
                    return self.send({"ok": True, **use_folder_stl(req.get("name"))})
            if u.path == "/api/cases/move":
                req = json.loads(self.body())
                return self.send({"ok": True, "name": move_case(req.get("case"), req.get("folder") or "")})
            if u.path == "/api/cases/rename":
                req = json.loads(self.body())
                return self.send({"ok": True, "name": rename_case(req.get("case"), req.get("new"))})
            if u.path == "/api/setup":
                return self.send({"ok": True, **setup_case(json.loads(self.body()))})
            if u.path == "/api/preview":
                return self.send({"ok": True, **preview_geometry(json.loads(self.body()))})
            if u.path == "/api/case_template":
                req = json.loads(self.body())
                return self.send({"ok": True, **case_template(req.get("case"), bool(req.get("keep_name")))})
            if u.path == "/api/case_params":
                req = json.loads(self.body())
                return self.send({"ok": save_case_params(req.get("case"), req.get("params") or {})})
            if u.path == "/api/cases/delete":
                req = json.loads(self.body())
                path = case_path(req.get("case"))
                name = case_rel(req.get("case"))
                if case_busy(name):
                    return self.send({"ok": False, "error": busy_error(name)})
                shutil.rmtree(path)
                return self.send({"ok": True})
            if u.path == "/api/queue/remove":
                qid = int(json.loads(self.body()).get("id", -1))
                with QUEUE_LOCK:
                    QUEUE[:] = [item for item in QUEUE if item["id"] != qid]
                return self.send({"ok": True})
            if u.path == "/api/restart":
                self.send({"ok": True})   # answered before the re-exec can cut the connection
                threading.Thread(target=restart_server, daemon=True).start()
                return
            if u.path == "/api/queue/clear":
                with QUEUE_LOCK:
                    QUEUE.clear()
                return self.send({"ok": True})
            if u.path == "/api/run":
                req = json.loads(self.body())
                a = req.get("action")
                if a == "stop":
                    stop_job()
                    return self.send({"ok": True})

                mcfg = req.get("mesh") or {}
                mesh_kwargs = {
                    "auto_retry": bool(mcfg.get("auto_retry", False)),
                    "max_retries": int(mcfg.get("max_retries", 2)),
                    "level_bump": int(mcfg.get("level_bump", 1)),
                }
                autostop_cfg = req.get("autostop")
                if not autostop_cfg or not autostop_cfg.get("enabled"):
                    autostop_cfg = None
                nprocs = int(req.get("nprocs", 1))
                if nprocs < 1:
                    return self.send({"ok": False, "error": "Number of processes must be at least 1."})

                if a == "paraview":
                    case = case_path(req.get("case"))
                    view = req.get("view") or "overview"
                    if view not in PV_VIEWS:
                        return self.send({"ok": False, "error": f"Unknown ParaView view '{view}'."})
                    near = str(req.get("near_mm") or "").strip()
                    try:
                        near = float(near) if near else None
                    except ValueError:
                        near = -1
                    if near is not None and not near > 0:
                        return self.send({"ok": False, "error": "Pathline distance must be a number of mm above 0, or blank."})
                    open(os.path.join(case, "pifoam_open.py"), "w").write(paraview_script(case, view, near))
                    # Ubuntu's ParaView package installs paraview.simple into dist-packages but its
                    # embedded Python doesn't search there ("No module named 'paraview'")
                    subprocess.Popen(foam_bash("touch case.foam; "
                                               "export PYTHONPATH=/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}; "
                                               "paraview --script=pifoam_open.py"),
                                     cwd=case, start_new_session=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return self.send({"ok": True})

                if a == "sweep":
                    param = req.get("param")
                    if param not in SWEEP_PARAMS:
                        return self.send({"ok": False, "error": f"Sweep parameter must be one of "
                                                                f"{', '.join(SWEEP_PARAMS)}."})
                    values = [SWEEP_PARAMS[param](str(v).strip()) for v in (req.get("values") or [])
                              if str(v).strip()]
                    if not values:
                        return self.send({"ok": False, "error": "Give at least one sweep value."})
                    if not STATE["verts"]:
                        return self.send({"ok": False, "error": "Choose an STL file first."})
                    base_params = dict(req.get("base") or {})
                    off = disabled_features(base_params)
                    if "studies" in off or (param in ("aoa", "yaw") and "angles" in off):
                        return self.send({"ok": False, "error": "That feature is switched off for this case. Switch "
                                                                "Studies (Studies heading) or Angle of attack / "
                                                                "yaw (Physics) On to use it."})
                    case_name = case_rel(base_params.get("case"))
                    ahead = run_job("sweep", case_name, do_sweep, base_params, param, values, mesh_kwargs,
                                    autostop_cfg, nprocs, STATE["verts"], max(1, int(req.get("avg") or AVG_WINDOW)))
                    return self.send({"ok": True, "queued": ahead})

                case = case_path(req.get("case"))
                case_name = case_rel(req.get("case"))
                if a == "sample":
                    g = case_geometry(case)
                    if not g:
                        return self.send({"ok": False, "error": "This case has no geometry and settings to sample."})
                    spec = {"probe_lines": req.get("probe_lines") or "", "probe_npts": req.get("probe_npts") or 100}
                    lines = parse_probes(spec, g)
                    if not lines["lines"]:
                        return self.send({"ok": False, "error": "Set at least one sample line under Run → Probes."})
                    outside = [ln["name"] for ln in lines["lines"] if not ln["inside"]]
                    if outside:
                        return self.send({"ok": False, "error": f"{', '.join(outside)} lie{'s' if len(outside) == 1 else ''} "
                                                                "outside the flow domain."})
                    ahead = run_job("sample", case_name, do_sample, case, lines["lines"], lines["npts"])
                    save_case_params(case_name, spec, keys=("probe_lines", "probe_npts"))
                    return self.send({"ok": True, "queued": ahead})
                if a == "mesh":
                    ahead = run_job("mesh", case_name, do_mesh, case, nprocs=nprocs, **mesh_kwargs)
                elif a == "solve":
                    ahead = run_job("solve", case_name, do_solve, case, autostop=autostop_cfg, nprocs=nprocs)
                elif a == "all":
                    ahead = run_job("all", case_name, do_all, case, mesh_cfg=mesh_kwargs,
                                    autostop_cfg=autostop_cfg, nprocs=nprocs)
                else:
                    return self.send({"ok": False, "error": f"Unknown action '{a}'."})
                return self.send({"ok": True, "queued": ahead})
        except Exception as e:
            return self.send({"ok": False, "error": str(e)})
        self.send({"error": "not found"}, code=404)


def startup_problems():
    """What stops this machine from running cases, each with its fix ([] = ready)."""
    if not os.path.isfile(BASHRC):
        return [f"OpenFOAM environment script not found: {BASHRC}\n"
                "  Install OpenFOAM 13, or set PIFOAM_BASHRC to its etc/bashrc."]
    out = []
    if not sh("command -v foamRun"):
        out.append(f"foamRun not found after sourcing {BASHRC} - is this OpenFOAM 13?")
    if not os.path.isdir(TUTORIAL):
        out.append(f"motorBike tutorial (the case template) not found: {TUTORIAL}\n"
                   "  Set PIFOAM_TUTORIAL to incompressibleFluid/motorBike/motorBike of an OpenFOAM 13 install.")
    return out


if __name__ == "__main__":
    problems = startup_problems()
    if problems:
        sys.exit("PIFOAMUI can't start:\n" + "\n".join(f"- {m}" for m in problems))
    print(f"OpenFOAM GUI running. Open http://localhost:{PORT} in your browser. Cases go in {RUN}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
