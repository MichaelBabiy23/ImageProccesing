# Conclusion: Lessons Learned from Pipeline Comparison

Student ID: 323073734

---

## What We Compared

We compared our classic CV pipeline against a stronger reference pipeline built around the same broad stages: contrast enhancement, edge-preserving denoising, adaptive thresholding, morphological cleanup, watershed separation, and skeletonization.

The main lesson from the final debugging pass is that the largest remaining errors were not in one isolated threshold, but in how several cleanup operations interacted after segmentation.

---

## Final Debugging Update: What Stage 8 Really Was

At first, `08_small_removed.png` looked like a single "remove small components" step. After splitting the pipeline into explicit substeps, we found that stage 8 was actually a composite cleanup block:

1. `08a_after_baseline_cut`
2. `08b_after_substrate_removed`
3. `08c_after_top_removed`
4. `08d_after_edge_removed`
5. `08e_after_bottom_artifact_removed`
6. `08_small_removed` (the final connected-component size filter)

This changed the diagnosis completely. On the difficult Hard images, the original problem was not just "small components." The real failure mode was:

- the intensity-based substrate removal sometimes climbed too high and cut through real roots
- the old edge filter also removed legitimate side branches near the left and right margins
- after those cuts disconnected the tree, the final size filter removed many of the remaining fragments

So the last visible loss in `08_small_removed.png` was often caused by mistakes that happened earlier inside the same stage.

---

## Easy Images

For the Easy set, the baseline and substrate logic is still useful, but it had to be constrained more tightly.

The current behavior is:

- `08a_after_baseline_cut` removes the bright band below the dendrite base
- `08b_after_substrate_removed` now removes only a narrow strip above that baseline

After the latest fix, substrate removal is capped to at most 8 rows above the detected baseline. In practice, on the Easy images it now trims roughly 2-8 rows above the baseline instead of 13-19 rows as before. This preserves more of the short dendrite stumps while still removing the bright substrate/interface band.

This means the Easy images now fail less by over-cutting the base and more by the usual issue: the finest branches are still fragile after segmentation and cleanup.

---

## Hard Images

The Hard `Ag_40nm_pitch_*` images were the most informative part of the analysis.

### What was going wrong

On several images, especially `Ag_40nm_pitch_001`, `002`, `004`, and `006`, the intensity-based substrate detector interpreted the brighter lower half of the SEM image as substrate and removed too much of the actual tree. In parallel, the old side filter treated real terminal branches near the margins as edge noise.

### What changed

We added three guardrails:

1. The intensity-based substrate cutoff is now accepted only when it stays low enough in the image.
2. The candidate bottom band must be visibly sparser than the region above it.
3. If a baseline is known, substrate removal cannot climb more than 8 rows above it.

We also made edge cleanup much stricter:

- only tiny, compact components that actually touch the image border are removed
- components merely near the side margins are kept

### What remains difficult

These fixes stopped the obvious over-cutting in `Ag_40nm_pitch_001` and similar cases. The lower arbor and side branches are now preserved much better in the intermediate outputs.

The main remaining weakness on Hard images is now the final component filter in `08_small_removed`, especially when the segmentation is already fragmented. `Ag_40nm_pitch_005` is still a clear failure case: it is no longer cut for the same substrate reason, but the final mask is still too sparse because many surviving fragments remain too small or too disconnected.

The `70nm_*` images remain structurally difficult for a different reason: periodic background patterns can still look too similar to dendrites at the pixel level.

---

## What We Changed and Why

### Stable improvements that remain important

1. `ADAPTIVE_C = -12`
   The most important segmentation choice. It selects bright dendrites directly instead of forcing a polarity inversion from the negative space.

2. Larger local context for adaptive thresholding
   A larger block size gives a more stable estimate of the local mean on SEM images with slow intensity drift.

3. Gentler denoising
   Lower CLAHE amplification, one bilateral pass, and no Gaussian blur preserve more branch edges before thresholding.

4. Reconstruction fallback
   If morphological reconstruction removes too much foreground, the pipeline now falls back to the pre-reconstruction mask instead of forcing the loss.

### Stage-8 fixes from the final analysis

5. Removed opening and closing in the `classic_no_open_close` variant
   These morphology steps were erasing real thin branches before the more targeted cleanup logic had a chance to operate.

6. Lowered `MIN_COMPONENT_AREA` from 150 to 90
   This was needed after we saw that many legitimate terminal fragments were being deleted only because earlier cleanup had disconnected them.

7. Made substrate removal conservative
   The intensity-based substrate detector now has explicit safety checks and is clamped relative to the detected baseline.

8. Tightened edge cleanup
   The filter now removes only border-touching compact specks, not any small object near the left or right side.

9. Exposed every stage-8 substep as a saved image
   This was crucial. Once stage 8 was decomposed, the real source of the errors became obvious.

---

## What Still Separates Us from the Reference

The reference pipeline is still denser on very fine tips and fragmented side branches. The remaining gap is now mostly:

1. Better preservation of disconnected thin fragments
   Their post-processing is still more tolerant of very small but meaningful branch pieces.

2. Better handling of difficult root regions
   Even after the substrate fix, our final cleanup can still be too aggressive when the lower tree is already fragmented.

3. Better robustness on periodic Hard backgrounds
   The `70nm_*` images still contain repeating structures that are difficult to reject using purely local morphology and thresholding.

---

## Summary

The final conclusion is different from what we first thought.

The biggest issue was not simply that "small-component removal was too aggressive." The deeper problem was that stage 8 bundled several cleanup operations together under one filename, which hid where the real damage was happening.

After splitting stage 8 and adding guards:

- substrate removal no longer cuts far above the baseline
- side branches near the image margins are no longer deleted as edge noise
- the pipeline is much easier to inspect and debug

The current bottleneck is now clearer: on difficult Hard images, the final connected-component filtering is still too destructive once the dendrite tree becomes fragmented. That is a more precise and more useful conclusion than the earlier assumption that the whole post-processing block was uniformly wrong.
