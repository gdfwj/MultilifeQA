# MultiLifeQA

## Download and Usage  

The `.zip` package contains the complete QA dataset.  
- The **original** folder provides all raw questions and answers.  
- The **simple** folder provides QA pairs under **Context Prompting**.  
- The **sql** folder provides QA pairs under **Database-augmented Prompting**.  

The dataset (≈300 MB) is hosted via **Git LFS** in this repository.  
Clone with LFS enabled to obtain the `.zip` file:

```bash
git lfs install
```

If Git LFS bandwidth is exceeded or unavailable, you can alternatively download the dataset directly from the following anonymous link: https://osf.io/dvrbz/?view_only=0e6eb442abf64aaaa6c504200b6140f6

Each folder contains multiple subfolders, where the folder name indicates the table category.  

### Folder Structure and Description  

- **activity_food_joint**: Joint tables combining **activity** and **diet** domains (single-user, cross-domain).  
- **all_joint_multi**: All domains combined into a joint multi-table setting (**multi-user, cross-domain**).  
- **heart_rate_variability**: Single table from the **sleep** domain (single-user).  
- **pa_active_minutes_multi**: Active minutes table from the **activity** domain (**multi-user**).  
- **pa_reports_multi**: Reports table from the **activity** domain (**multi-user**).  
- **skin_temp_sleep_nightly_multi**: Nightly skin temperature table from the **sleep** domain (**multi-user**).  
- **activity_sleep_joint**: Joint tables combining **activity** and **sleep** domains (single-user, cross-domain).  
- **emotion_food_joint**: Joint tables combining **emotion** and **diet** domains (single-user, cross-domain).  
- **heart_rate_variability_multi**: Heart rate variability table from the **activity** domain (**multi-user**).  
- **pa_daily_summary**: Daily summary table from the **activity** domain (single-user).  
- **physical_activity_joint**: Joint tables within the **activity** domain (single-user, multi-table).  
- **sleep_joint**: Joint tables within the **sleep** domain (single-user, multi-table).  
- **additional_sleep**: Additional table from the **sleep** domain (single-user).  
- **food_meal_labels**: Meal label table from the **diet** domain (single-user).  
- **oxygen_sat_daily**: Daily oxygen saturation table from the **activity** domain (single-user).  
- **pa_daily_summary_multi**: Daily summary table from the **activity** domain (**multi-user**).  
- **respiratory_rate**: Respiratory rate table from the **sleep** domain (single-user).  
- **sleep_stress_joint**: Joint tables combining **sleep** and **emotion** domains (single-user, cross-domain).  
- **additional_sleep_multi**: Additional table from the **sleep** domain (**multi-user**).  
- **food_meal_labels_multi**: Meal label table from the **diet** domain (**multi-user**).  
- **oxygen_sat_daily_multi**: Daily oxygen saturation table from the **sleep** domain (**multi-user**).  
- **pa_emotion_joint**: Joint tables combining **activity** and **emotion** domains (single-user, cross-domain).  
- **respiratory_rate_multi**: Respiratory rate table from the **activity** domain (**multi-user**).  
- **stress_daily_scores**: Daily stress scores table from the **emotion** domain (single-user).  
- **all_joint**: All domains combined into a joint multi-table setting (single-user, cross-domain).  
- **food_sleep_joint**: Joint tables combining **diet** and **sleep** domains (single-user, cross-domain).  
- **pa_active_minutes**: Active minutes table from the **activity** domain (single-user).  
- **pa_reports**: Reports table from the **activity** domain (single-user).  
- **skin_temp_sleep_nightly**: Nightly skin temperature table from the **sleep** domain (single-user).  
- **stress_daily_scores_multi**: Daily stress scores table from the **emotion** domain (**multi-user**).  

Each subfolder contains five `.jsonl` files: **FQ**, **AS**, **CQ**, **NC**, and **TA**, where each line corresponds to a QA pair.  

---

## Evaluating Context Prompting  

The `simple` folder stores QA pairs generated with Context Prompting.  
- `"Query"`: the question.  
- `"Answer"`: the ground-truth answer.  

Run our evaluation with:  

```bash
python eval_simple.py \
  --data-root ./gen_data_processed/simple \
  --eval-root ./eval \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --max-new-tokens 32 \
  --api-key  "" 
```

## Evaluating Database-augmented Prompting

### Build the MySQL Database

1. Create a folder `data`.
2. Download [AI4FoodDB](https://github.com/AI4Food/AI4FoodDB) and place the `dataset/` contents inside `data/`.
3. Download [FoodNExtDB.zip](https://bidalab.eps.uam.es/static/AI4FoodDB/FoodNExtDB.zip), extract it, and move into `data/FoodNExtDB/`.

The structure should look like:

```
- data
  -DS1_AnthropometricMeasurements
  -DS2_LifestyleHealth
  ...
  -DS10_AdditionalInformation
  -participant_information.csv
  -FoodNExtDB
    -A4F_10021
    ...
    -A4F_99000
- schema.sql
- load_mysql_db.py
- load_food_db.py
```

1. Install MySQL and create a database named `MultilifeQA`.
2. Update the `user` fields in `load_mysql_db.py` and `load_food_db.py` to your local settings, then run both scripts.

### Run Evaluation

Each `.jsonl` file in the `sql` folder contains the following fields:
- `"Query_sql"`: the base question for SQL generation, where constraints need to be added.  
- `"Query_base"`: the base question for generating the final answer from the executed SQL results, where the specific SQL and result should be appended.  
- `"Answer"`: the ground-truth answer.  

For usage examples, please refer to `eval_sql.py`.

Run our evaluation with:  

```
python eval_sql.py \
  --data-root ./gen_data_processed/sql \
  --eval-root ./eval_sql \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --sql-max-new-tokens 480 \
  --ans-max-new-tokens 48 \
  --api-key  ""
```
