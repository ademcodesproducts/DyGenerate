# analyze.py
#
# What it is for:
#   Turns results/scores.csv into the Exp 1 numbers and plots that decide Gate A.
#
# What to include:
#
# 1. Accuracy per model x generator x cell x prompt version, averaging samples within an item first.
#
# 2. The three item sets per model
#    - G1, fresh in range: home generator, inside the training range (what dynamic benchmarks report today)
#    - G2: home generator, beyond the training range
#    - G4: away generator at matched settings
#    - Gold accuracy = G2 and G4 pooled.
#
# 3. Home advantage per model: accuracy on the home generator minus accuracy on the away generator,
#    at the same board size or number count and difficulty, inside the training range.
#
# 4. Difference in differences: home advantage of the exposed model minus that of its parent,
#    and minus that of each broad model. This cancels a gap caused by one generator simply being harder.
#
# 5. Uncertainty: 95% bootstrap intervals, resampling items (all samples of an item move together).
#    "No difference" claims use an equivalence test at plus or minus 3 points, never non-significance.
#
# 6. Rankings: rank models by G1 and by gold accuracy; list every pair that swaps order (first H2 evidence).
#
# 7. Native versus canonical: the home advantage under each prompt version, to separate wording from puzzle structure.
#
# 8. Tables of truncation and no-answer rates per model and cell.
#
# 9. Plots saved to results/: accuracy against board size (Futoshiki) and number count (Countdown),
#    one line per model, home and away side by side, the training range shaded.
#
# 10. A short written summary saved to results/: is there a home advantage beyond the parent's, how large, and how sure.
