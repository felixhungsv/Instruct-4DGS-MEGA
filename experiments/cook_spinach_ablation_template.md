# cook_spinach Baseline vs Hybrid Report

## Experiment Setup
- Dataset: `dynerf/cook_spinach`
- Prompts:
  - `Make it look like a fauvism painting`
  - `Make it look like a sculpture`
  - `Turn the man into a woman`
- Baseline command: `bash script.sh dynerf cook_spinach 10.5 1.2 sh 0.0`
- Hybrid command: `bash script.sh dynerf cook_spinach 10.5 1.2 lite 0.002`

## Compare Metrics
| Prompt | Variant | Runtime (sec) | total_mb | packed_fp16.zip_mb | Video path |
|---|---:|---:|---:|---:|---|
| fauvism painting | baseline |  |  |  |  |
| fauvism painting | hybrid |  |  |  |  |
| sculpture | baseline |  |  |  |  |
| sculpture | hybrid |  |  |  |  |
| man->woman | baseline |  |  |  |  |
| man->woman | hybrid |  |  |  |  |

## Notes
- Visual quality differences (detail, flicker, color stability):  
- Memory/storage gain:  
- Suggested next tuning (entropy weight, guidance, iteration):  
