# Baseline results

[model_comparison.csv](model_comparison.csv) contains the actual saved validation summary from Dario's notebook. Test accuracy and inference time are empty because neither has been measured in the supplied work.

Trainable parameter counts are derived from the model definitions. Total parameters, checkpoint epochs, validation accuracies, and training times come from saved outputs. Accuracy values are fractions.

Training-time measurements include validation and artifact-saving overhead; they are not inference timings. See [the results document](../docs/training_results.md) for details.

Add loss/accuracy curves and the chosen model's confusion matrix to figures/ when the source logs and test evaluation are available.
