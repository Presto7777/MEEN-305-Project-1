"""Reports and visuals for saved designs (DesignSpec): analyze one, compare several.

Needs matplotlib.  Caller chooses the backend (use matplotlib.use("Agg") for headless).
"""
import os
from dataclasses import replace
import matplotlib.pyplot as plt

from .design import DesignSpec, ORIENTATIONS
from .loads import standard_supports, create_load_case
from .failure import failure_table, failure_summary
from .report import model_summary, result_summary, export_csv
from . import viz


def _sens_table(spec, K_values, knockdowns):
    lines = ["Sensitivity of minimum FS (over C and D):",
             "  print orientation x effective-length factor K"]
    lines.append("  " + f"{'orientation':<12}" + "".join(f"K={k:<7g}" for k in K_values))
    for o in ORIENTATIONS:
        row = [replace(spec, orientation=o, K=k).evaluate().min_fs for k in K_values]
        lines.append("  " + f"{o:<12}" + "".join(f"{v:<9.2f}" for v in row))
    lines.append("  strength knockdown (all strengths x factor)")
    lines.append("  " + "".join(f"{k:<9g}" for k in knockdowns))
    row = []
    for kd in knockdowns:
        mat = replace(spec.material, strength=replace(spec.material.strength, knockdown=kd))
        row.append(replace(spec, material=mat).evaluate().min_fs)
    lines.append("  " + "".join(f"{v:<9.2f}" for v in row))
    return "\n".join(lines)


def analyze_design(spec: DesignSpec, outdir, K_values=(0.7, 1.0, 1.5),
                   knockdowns=(1.0, 0.8, 0.6), sensitivity=True) -> str:
    """Re-run a saved design and write plots, CSVs and summary.txt into `outdir`.
    Returns the summary text."""
    os.makedirs(outdir, exist_ok=True)
    ev = spec.evaluate()
    model, sections = ev.model, ev.sections
    sup = standard_supports(model)
    txt = [f"Design: {spec.name or '(unnamed)'}   layout {spec.layout}, span {spec.span:g}, "
           f"height {spec.height:g}, width {spec.depth:g}, panels {spec.panels}",
           f"orientation {spec.orientation}, K {spec.K:g}, required FS {spec.fs_required:g}, "
           f"load {spec.mass_kg:g} kg per condition",
           *[f"  {g:<5} x{ev.group_counts[g]:<3} {ev.section_desc[g]}" for g in ev.section_desc],
           "", model_summary(model, sections, spec.material), ""]
    for cond in "CD":
        res, fr = ev.results[cond], ev.failures[cond]
        txt += [result_summary(model, res), "", failure_summary(fr), "",
                failure_table(model, fr, top=10), ""]
    txt += [str(ev.check), ""]
    cap = spec.mass_kg * ev.min_fs / spec.fs_required
    txt += [f"Load margin: because the model is linear, FS scales as 1/load, so this design "
            f"meets FS {spec.fs_required:g} up to about {cap:.1f} kg per load point "
            f"(analysed: {spec.mass_kg:g} kg).", ""]
    if sensitivity:
        txt += [_sens_table(spec, K_values, knockdowns), ""]

    for cond in "CD":
        lc = create_load_case(model, cond, sup, spec.mass_kg)
        res = ev.results[cond]
        fig = viz.inspect_model(model, res, sup, lc, path=os.path.join(outdir, f"overview_{cond}.png"))
        plt.close(fig)
        fig = plt.figure(figsize=(7, 5))
        viz.plot_deformed_3d(model, res, ax=fig.add_subplot(projection="3d"))
        fig.savefig(os.path.join(outdir, f"deformed_{cond}.png"), dpi=130); plt.close(fig)
        export_csv(model, res, os.path.join(outdir, f"results_{cond}"), sections)
    fig = plt.figure(figsize=(13, 5))
    viz.plot_safety_factors(model, ev.envelope, spec.fs_required, "3d",
                            fig.add_subplot(1, 2, 1, projection="3d"), "Worst FS over C and D")
    viz.plot_safety_factors(model, ev.envelope, spec.fs_required, "front", fig.add_subplot(1, 2, 2))
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "safety_factors.png"), dpi=130); plt.close(fig)

    spec.save(os.path.join(outdir, "design.json"), ev)
    text = "\n".join(txt)
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write(text)
    return text


def compare_designs(specs, labels=None, path="compare.png"):
    """Evaluate several specs; return (table text, list of DesignEvaluation) and save a
    mass-vs-min-FS scatter plot to `path`."""
    labels = labels or [s.name or f"design {i + 1}" for i, s in enumerate(specs)]
    evs = [s.evaluate() for s in specs]
    rows = [f"{'design':<28}{'layout':<11}{'pan':>4}{'mass[g]':>9}{'min FS':>8}  "
            f"{'governing':<22}{'uz_D[mm]':>9}  status"]
    for lab, s, e in zip(labels, specs, evs):
        c = e.check
        rows.append(f"{lab[:27]:<28}{s.layout:<11}{s.panels:>4}{e.mass:9.2f}{c.min_fs:8.2f}  "
                    f"{('m' + str(c.member_id) + ' ' + c.mode)[:21]:<22}{e.max_uz['D']:9.2f}  "
                    f"{'PASS' if c.passes else 'FAIL'}")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    req = specs[0].fs_required
    ymax = max(10.0, 2 * max(e.min_fs for e in evs))
    ax.axhspan(req, ymax, color="#27ae60", alpha=0.10)
    ax.axhline(req, color="#27ae60", ls="--", lw=1.5)
    for lab, e in zip(labels, evs):
        ax.scatter([e.mass], [e.min_fs], s=90, zorder=3, edgecolor="black",
                   color="#27ae60" if e.check.passes else "#e74c3c")
        ax.annotate(lab, (e.mass, e.min_fs), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("ideal member mass [g]"); ax.set_ylabel("minimum FS (log)")
    ax.set_title("Saved designs: min safety factor vs. mass"); ax.grid(alpha=0.3, which="both")
    ax.set_xlim(left=0)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return "\n".join(rows), evs