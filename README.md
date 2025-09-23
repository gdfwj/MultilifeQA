# MultiLifeQA

## Download and Usage  

The `gen_data_processed` folder package contains the complete QA dataset.  
- The **original** folder provides all raw questions and answers.  
- The **simple** folder provides QA pairs under **Context Prompting**.  
- The **sql** folder provides QA pairs under **Database-augmented Prompting**.  

The dataset is hosted via **Git LFS** in this repository.  
Clone with LFS enabled to obtain the `gen_data_processed` folder:

```bash
git lfs install
```

If Git LFS bandwidth is exceeded or unavailable, you can alternatively download the `.zip` package of dataset directly from the following anonymous link: https://osf.io/download/68d23cac87759129021e4648/?view_only=6b74757f20be4b2b8d1db5c9b5e9d551

Each folder contains multiple subfolders, where the folder name indicates the table category.  
**In addition, we provide three summary files: `all_prompts.jsonl` (all questions) is placed in the root of the `original`, `simple`, and `sql` folders; and `single_user.jsonl` / `multi_user.jsonl` (single-user and multi-user question collections) are placed in the root of the `original` and `sql` folders.** 

### Folder Structure and Description  

We organize the health reasoning tasks in a hierarchical manner: **single-dimension → intra-domain multi-dimension → cross-domain multi-dimension**.  
- **single**: Single-dimension tasks within only one domain (diet, activity, sleep, or emotion). The folder name corresponds to the table name.  
- **activity_joint, sleep_joint**: Since activity and sleep domains contain multiple fine-grained indicators stored in separate relational tables, we design intra-domain multi-dimension reasoning tasks (M-activity, M-sleep).  
- **\*_\*_joint**: Cross-domain multi-dimension reasoning tasks, combining tables from two different domains (e.g., activity + sleep).  
- **all_joint**: Comprehensive reasoning tasks across all four domains, which is the most challenging setup.  
- **\*_multi**: Indicates the **multi-user** setting of the corresponding table.  

#### Single (single-dimension tasks)  
- additional_sleep  
- additional_sleep_multi  
- food_meal_labels  
- food_meal_labels_multi  
- pa_active_minutes  
- pa_active_minutes_multi  
- pa_daily_summary  
- pa_daily_summary_multi  
- pa_reports  
- pa_reports_multi  
- respiratory_rate  
- respiratory_rate_multi  
- skin_temp_sleep_nightly  
- skin_temp_sleep_nightly_multi  
- stress_daily_scores  
- stress_daily_scores_multi  
- heart_rate_variability  
- heart_rate_variability_multi  
- oxygen_sat_daily  
- oxygen_sat_daily_multi  

#### Activity Joint (intra-domain multi-dimension)  
- physical_activity_joint  

#### Sleep Joint (intra-domain multi-dimension)  
- sleep_joint  

#### Cross-domain Joint (\*_\*_joint)  
- activity_food_joint  
- activity_sleep_joint  
- emotion_food_joint  
- food_sleep_joint  
- pa_emotion_joint  
- sleep_stress_joint  

#### All Joint (cross all four domains)  
- all_joint  
- all_joint_multi  


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


## How to Build Your Own Dataset

We provide not only the processed dataset, but also a **generalizable pipeline** for automatic QA generation.  
If you are interested in exploring **new reasoning problems** or **customized question types**, you can easily extend our framework:

1. **Prepare and Load Your Data into MySQL**  
   - First, organize your dataset into structured CSV or relational tables.  
   - Then, follow the workflow in `load_mysql.py`:  
     - Check how the constructor and helper functions define table names, data paths, and schema mappings.  
     - Adapt these functions for your own dataset by modifying the corresponding names and paths.  
   - Once adjusted, run the script to load your dataset into MySQL, which ensures compatibility.  

2. **Apply your own template**  
   - Define the question style and reasoning constraints you want to study.  
   - You can create new templates under the same schema (FQ, AS, CQ, NC, TA), or design entirely new categories.  
   - Once the template is prepared, simply replace the corresponding part in provided sample `build.py`. 

We encourage the community to:  
- Extend to **new reasoning challenges** (e.g., more complex questions, longitudinal trend analysis, causal reasoning).  
- Explore **new prompting strategies** beyond Context and Database-augmented Prompting.  
- Share and compare results on customized datasets built upon our framework.  

By doing so, you can continually push the boundary of how LLMs reason over complex, multi-domain data.


