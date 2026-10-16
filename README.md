# PIFOAMUI

A browser GUI for running [OpenFOAM 13](https://openfoam.org) external-flow simulations around an STL part.
Upload a part, set the flow, mesh, solve and read the force coefficients without writing a single dictionary.

It is **one Python file** with no dependencies beyond the Python 3 standard library: it serves its own
web page (no CDN, no build step) and drives OpenFOAM through the shell.

## Features

- **Cases and folders**: create, open, rename, move (drag and drop) and delete cases in nested folders
  under `$FOAM_RUN`; a folder can hold a shared part for all its cases; "use case as template".
- **Physics**: fluid library, speed and direction, angle of attack / yaw, fixed or moving wind-tunnel
  ground, half models on a symmetry plane, Spalart–Allmaras, k-ω SST and γ-Reθ transition turbulence models,
  steady (SIMPLE) or transient (PIMPLE, Courant-limited time step, optional warm start).
- **Meshing**: blockMesh + snappyHexMesh with presets or explicit cell sizes, boundary layers sized by
  relative thickness or a y+ target (with a first-layer calculator), a wake refinement box, a configurable
  domain, a checkMesh quality report and automatic retry at a finer level when the mesh fails.
- **Running**: parallel (MPI) meshing and solving, a FIFO job queue, auto-stop on convergence, progress
  with ETA, graceful stop that still writes the fields.
- **Results**: Cd/Cl/Cm and forces/moments averaged over the last N iterations (time-weighted for
  transient runs, with a Strouhal number), convergence and residual plots, inlet/outlet monitors, point
  probes and sample lines, comparing two cases, CSV export and a printable HTML report, preset ParaView views.
- **Studies**: parameter sweeps (speed, angles, mesh resolution), mesh-independence studies, aircraft
  polars (CLα, CLmax, best L/D, Oswald e), neutral point / static margin, flight speeds from the mass,
  yaw-stability derivatives.
- **UI**: Basic / Intermediate / Advanced modes, hover help on every control, 3-D part and domain
  previews, unsaved settings kept per case in the browser.

## Requirements

- Linux, or Windows with **WSL2** (Ubuntu)
- **OpenFOAM 13** (the openfoam.org / Foundation release, e.g. the `openfoam13` Ubuntu package)
- Python 3 (comes with Ubuntu)
- Optional: ParaView, for viewing fields (`sudo apt install paraview`)

## Quick start

```bash
python3 PIFOAMUI.py
```

Then open <http://localhost:8080> in your browser (on Windows, the WSL server is reachable from the Windows
browser at the same address). Cases are written to `$FOAM_RUN`.

If OpenFOAM isn't where the script expects it, it tells you what's missing and exits.

## Configuration

All optional, set as environment variables:

| Variable          | Default                                                     | Meaning                               |
|-------------------|-------------------------------------------------------------|---------------------------------------|
| `PIFOAM_BASHRC`   | `/opt/openfoam13/etc/bashrc`                                | OpenFOAM 13's environment script      |
| `PIFOAM_TUTORIAL` | `$FOAM_TUTORIALS/incompressibleFluid/motorBike/motorBike`   | Tutorial case used as the case template |
| `PIFOAM_PORT`     | `8080`                                                      | HTTP port                             |

```bash
PIFOAM_BASHRC=$HOME/OpenFOAM/OpenFOAM-13/etc/bashrc PIFOAM_PORT=8081 python3 PIFOAMUI.py
```

## Security

The server listens on `127.0.0.1` only, has **no authentication**, and runs shell commands on your
machine. Don't expose it to a network (port forwarding, reverse proxies, `0.0.0.0`).

## Disclaimer

Simulation results depend on your mesh, settings and judgement; check them before relying on them.
The software comes with no warranty (see the license).

This project is not approved or endorsed by OpenCFD Ltd, ESI Group or the OpenFOAM Foundation.
OPENFOAM® is a registered trademark of OpenCFD Ltd.

## Credits

- Several features are ported from [Magnusim](https://github.com/Lilmill2000/Magnusim) (GPL-3.0-or-later).
- Fonts: [IBM Plex](https://github.com/IBM/plex) Sans Condensed and Mono, embedded under the SIL Open Font License 1.1.

## License

[GPL-3.0-or-later](LICENSE).
