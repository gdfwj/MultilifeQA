/* =========================================================
 *  Database: AI4food_db
 *  DDL for participant-centred multi-modal dataset
 *  (c) 2025
 * ========================================================= */

-- 0. 建库
CREATE DATABASE IF NOT EXISTS `MultilifeQA`
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `MultilifeQA`;

/* ---------------------------------------------------------
 * 1. 核心表
 * --------------------------------------------------------- */
DROP TABLE IF EXISTS `participants`;
CREATE TABLE `participants` (
  `id`                     VARCHAR(20)  NOT NULL,
  `group`                  INT,
  `age`                    INT,
  `sex`                    ENUM('Male','Female','Other') DEFAULT NULL,
  `usual_weight_kg`        DECIMAL(5,2),
  `weight_5years_kg`       DECIMAL(5,2),
  `height_cm`              DECIMAL(5,2),
  `intervention_diet_kcal` SMALLINT UNSIGNED,
  `finished_intervention`  TINYINT(1),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* ---------------------------------------------------------
 * 2. DS1 – Anthropometric measurements
 * --------------------------------------------------------- */
-- DROP TABLE IF EXISTS `anthropometrics`;
-- CREATE TABLE `anthropometrics` (
--   `id`                 VARCHAR(20) NOT NULL,
--   `visit`              TINYINT UNSIGNED NOT NULL,
--   `period`             VARCHAR(20),
--   `current_weight_kg`  DECIMAL(5,2),
--   `bmi_kg_m2`          DECIMAL(5,2),
--   `fat_mass_perc`      DECIMAL(5,2),
--   `muscle_mass_perc`   DECIMAL(5,2),
--   `visceral_fat_level` DECIMAL(4,1),
--   `basal_metabolism`   SMALLINT UNSIGNED,
--   `waist_cm`           DECIMAL(5,2),
--   `hip_cm`             DECIMAL(5,2),
--   PRIMARY KEY (`id`,`visit`),
--   CONSTRAINT `fk_anthro_participant`
--     FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
--       ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- /* ---------------------------------------------------------
--  * 3. DS2 – Health & Lifestyle
--  * --------------------------------------------------------- */
-- DROP TABLE IF EXISTS `health`;
-- CREATE TABLE `health` (
--   `id`                  VARCHAR(20) NOT NULL,
--   `oral_contraceptive`  TINYINT(1),
--   `ring_contraceptive`  TINYINT(1),
--   `antidepressants`     TINYINT(1),
--   `antiacids`           TINYINT(1),
--   `antihistamines`      TINYINT(1),
--   `antiinflamatory`     TINYINT(1),
--   `iron`                TINYINT(1),
--   `calcium`             TINYINT(1),
--   `antihypertensives`   TINYINT(1),
--   `thyroid_hormone`     TINYINT(1),
--   `antibiotics`         TINYINT(1),
--   `other_medication`    TINYINT(1),
--   `no_medication`       TINYINT(1),
--   `hyperthyroidism`     TINYINT(1),
--   `hypothyroidism`      TINYINT(1),
--   `hypercholesterolemia`TINYINT(1),
--   `triglyceridemia`     TINYINT(1),
--   `hypertension`        TINYINT(1),
--   `depression`          TINYINT(1),
--   `diabetes`            TINYINT(1),
--   `lactose_intolerance` TINYINT(1),
--   `other_illness`       TINYINT(1),
--   `no_illness`          TINYINT(1),
--   `menopause`           TINYINT(1),
--   PRIMARY KEY (`id`),
--   FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
--     ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- DROP TABLE IF EXISTS `lifestyle`;
-- CREATE TABLE `lifestyle` (
--   `id`                      VARCHAR(20) NOT NULL,
--   `appetite`                BIGINT,
--   `daily_meals`             VARCHAR(20),
--   `meals_weekdays_out`      DECIMAL(6,2),
--   `meals_weekdays_home`     DECIMAL(6,2),
--   `meals_weekend_out`       DECIMAL(6,2),
--   `meals_weekend_home`      DECIMAL(6,2),
--   `defecation`              VARCHAR(20),
--   `urination`               VARCHAR(20),
--   `water_ml`                BIGINT,
--   `others_ml`               DECIMAL(6,2),
--   `cigarettes_day`          BIGINT,
--   `cigars_day`              BIGINT,
--   `pipe_day`                BIGINT,
--   `alcohol`                 VARCHAR(20),
--   `fermented_perc`          DECIMAL(6,2),
--   `distilled_perc`          DECIMAL(6,2),
--   `exercise`                VARCHAR(20),
--   `stress`                  TINYINT,
--   `anxiety`                 TINYINT,
--   `depression`              TINYINT,
--   `eating_disorder`         TINYINT,
--   `others_psychological`    TINYINT,
--   `no_psychological`        TINYINT,
--   `sleep_weekdays`          DECIMAL(6,2),
--   `sleep_weekend`           DECIMAL(6,2),
--   `insomnia`                TINYINT,
--   `somnolence`              TINYINT,
--   PRIMARY KEY (`id`),
--   FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
--     ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- /* ---------------------------------------------------------
--  * 4. DS4 – Biomarkers
--    # 注意小于号
--  * --------------------------------------------------------- */
-- DROP TABLE IF EXISTS `biomarkers`;
-- CREATE TABLE `biomarkers` (
--   `id`                    VARCHAR(20) NOT NULL,
--   `visit`                 TINYINT UNSIGNED NOT NULL,
--   `leukocytes_10e3_ul`    DECIMAL(6,2),
--   `plats_10e3_ul`         DECIMAL(6,2),
--   `lympho_10e3_ul`        DECIMAL(6,2),
--   `mono_10e3_ul`          DECIMAL(6,2),
--   `seg_10e3_ul`           DECIMAL(6,2),
--   `eos_10e3_ul`           DECIMAL(6,2),
--   `baso_10e3_ul`          DECIMAL(6,2),
--   `erythrocytes_10e6_ul`  DECIMAL(6,2),
--   `hgb_g_dl`              DECIMAL(5,2),
--   `hematocrit_perc`       DECIMAL(5,2),
--   `mcv_fl`                DECIMAL(5,2),
--   `mpv_fl`                DECIMAL(5,2),
--   `mch_pg`                DECIMAL(5,2),
--   `mchc_g_dl`             DECIMAL(5,2),
--   `rdw_perc`              DECIMAL(5,2),
--   `lympho_perc`           DECIMAL(5,2),
--   `mono_perc`             DECIMAL(5,2),
--   `seg_perc`              DECIMAL(5,2),
--   `eos_perc`              DECIMAL(5,2),
--   `baso_perc`             DECIMAL(5,2),
--   `hba1c_perc`            DECIMAL(5,2),
--   `hba1ifcc_mmol_mol`     DECIMAL(6,2),
--   `insulin_uui_ml`        DECIMAL(6,2),
--   `homa`                  DECIMAL(6,2),
--   `glu_mg_dl`             SMALLINT,
--   `chol_mg_dl`            SMALLINT,
--   `tri_mg_dl`             SMALLINT,
--   `hdl_mg_dl`             SMALLINT,
--   `ldl_mg_dl`             SMALLINT,
--   `homocysteine_umol_l`   DECIMAL(6,2),
--   `alb_g_dl`              DECIMAL(5,2),
--   `prealbumin_mg_dl`      DECIMAL(6,2),
--   `crp_mg_dl`             DECIMAL(6,2),
--   `igg_mg_dl`             DECIMAL(6,2),
--   `iga_mg_dl`             DECIMAL(6,2),
--   `igm_mg_dl`             DECIMAL(6,2),
--   `ige_ui_ml`             DECIMAL(8,2),
--   `tnf_a_ui_ml`           DECIMAL(8,2),
--   `adiponectin_ug_ml`     DECIMAL(8,2),
--   PRIMARY KEY (`id`,`visit`),
--   FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
--     ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- /* ---- Continuous glucose monitor (per-minute) ---- */
-- DROP TABLE IF EXISTS `glucose_levels`;
-- CREATE TABLE `glucose_levels` (
--   `id`            VARCHAR(20) NOT NULL,
--   `ts`            DATETIME    NOT NULL,
--   `glucose_value_in_mg_dl` SMALLINT UNSIGNED,
--   PRIMARY KEY (`id`,`ts`),
--   FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
--     ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- /* ---------------------------------------------------------
--  * 5. DS6 – Vital Signs
--  * --------------------------------------------------------- */
-- DROP TABLE IF EXISTS `vital_signs`;
-- CREATE TABLE `vital_signs` (
--   `id`               VARCHAR(20) NOT NULL,
--   `visit`            TINYINT UNSIGNED NOT NULL,
--   `systolic_blood_pressure_mmhg`   SMALLINT,
--   `diastolic_blood_pressure_mmhg` SMALLINT,
--   `heart_rate_bpm`   SMALLINT,
--   PRIMARY KEY (`id`,`visit`),
--   FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
--     ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- /* ---------------------------------------------------------
--  * ecg_recordings + ecg_waveforms
--  * --------------------------------------------------------- */
-- DROP TABLE IF EXISTS `ecg_recordings`;
-- CREATE TABLE `ecg_recordings` (
--   `id`                    VARCHAR(20)  NOT NULL,              -- 参与者 ID
--   `record_ts`             DATETIME     NOT NULL,              -- 本次 ECG 开始时间（CSV 的 timestamp）
--   `result_classification` VARCHAR(20),                        -- 如 NSR / AF 等
--   `average_heart_rate`    SMALLINT,                           -- 平均心率 (bpm)
--   `heart_rate_alert`      VARCHAR(20),                        -- NONE / HIGH / LOW ...
--   `sample_count`          INT UNSIGNED,                       -- 该波形共有多少采样点
--   PRIMARY KEY (`id`, `record_ts`),
--   FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
--     ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;
-- DROP TABLE IF EXISTS `ecg_waveforms`;
-- CREATE TABLE `ecg_waveforms` (
--   `id`          VARCHAR(20)  NOT NULL,
--   `record_ts`   DATETIME     NOT NULL,       -- 必须与 ecg_recordings.record_ts 对应
--   `sample_idx`  INT UNSIGNED NOT NULL,       -- 从 0 开始的采样点索引
--   `voltage`     SMALLINT,                    -- 电压值（ADC 计数 / μV）
--   PRIMARY KEY (`id`, `record_ts`, `sample_idx`),
--   FOREIGN KEY (`id`, `record_ts`) REFERENCES `ecg_recordings` (`id`, `record_ts`)
--     ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- /* ---------------------------------------------------------
--  * heart_rate
--  * --------------------------------------------------------- */
-- DROP TABLE IF EXISTS `heart_rate`;
-- CREATE TABLE `heart_rate` (
--   `id`  VARCHAR(20) NOT NULL,
--   `ts`  DATETIME    NOT NULL,
--   `bpm` SMALLINT,
--   PRIMARY KEY (`id`,`ts`),
--   FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
--     ON UPDATE CASCADE ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* ---------------------------------------------------------
 * 6. DS7 – Physical Activity
 * --------------------------------------------------------- */

/* ============  IPAQ（International Physical Activity Questionnaire） ============ */
DROP TABLE IF EXISTS `physical_activity_ipaq`;
CREATE TABLE `physical_activity_ipaq` (
  `id`                 VARCHAR(20)  NOT NULL,              -- 参与者 ID
  `visit`              TINYINT UNSIGNED NOT NULL,          -- 第几次随访

  `vigorous_n_days`    TINYINT,                            -- 每周做剧烈活动的天数
  `vigorous_min_day`   SMALLINT,                           -- 剧烈活动当日分钟数
  `vigorous_met`       SMALLINT,                           -- 剧烈活动 MET∙min/周

  `moderate_n_days`    TINYINT,
  `moderate_min_day`   SMALLINT,
  `moderate_met`       SMALLINT,

  `walking_n_days`     TINYINT,
  `walking_min_day`    SMALLINT,
  `walking_met`        SMALLINT,

  `total_met`          SMALLINT,                           -- 总 MET∙min/周
  `categorical_score`  ENUM('low','moderate','high'),       -- IPAQ 分类结果

  PRIMARY KEY (`id`,`visit`), 
  FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* ========= 1. 每日各强度分钟数 =========== */
DROP TABLE IF EXISTS `pa_active_minutes`;
CREATE TABLE `pa_active_minutes` (
  `id`                         VARCHAR(20) NOT NULL,
  `ts`                        DATETIME        NOT NULL,        -- 由 timestamp 取日期部分
  `fat_burn_minutes`           SMALLINT UNSIGNED,
  `cardio_minutes`             SMALLINT UNSIGNED,
  `peak_minutes`               SMALLINT UNSIGNED,
  `sedentary_minutes`          SMALLINT UNSIGNED,
  `lightly_active_minutes`     SMALLINT UNSIGNED,
  `moderately_active_minutes`  SMALLINT UNSIGNED,
  `very_active_minutes`        SMALLINT UNSIGNED,
  PRIMARY KEY (`id`,`ts`),
  CONSTRAINT `fk_pa_intensity_id`
      FOREIGN KEY (`id`) REFERENCES `participants`(`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* ========= 2. 每日心率区间 / 计步汇总 =========== */
DROP TABLE IF EXISTS `pa_daily_summary`;
CREATE TABLE `pa_daily_summary` (
  `id`                              VARCHAR(20) NOT NULL,
  `ts`                             DATETIME        NOT NULL,
  `resting_heart_rate`              DECIMAL(5,2),      -- bpm
  `altitude_m`                      SMALLINT,          -- 样例中整数
  `calories_kcal`                   DECIMAL(8,2),
  `steps`                           INT UNSIGNED,
  `distance_m`                      DECIMAL(10,2),
  `minutes_below_default_zone_1`        SMALLINT UNSIGNED,
  `minutes_in_default_zone_1`           SMALLINT UNSIGNED,
  `minutes_in_default_zone_2`           SMALLINT UNSIGNED,
  `minutes_in_default_zone_3`           SMALLINT UNSIGNED,
  PRIMARY KEY (`id`,`ts`),
  CONSTRAINT `fk_pa_sum mary_id`
      FOREIGN KEY (`id`) REFERENCES `participants`(`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* ========= 3. VO2Max 每日估计 =========== */
DROP TABLE IF EXISTS `pa_estimated_VO2`;
CREATE TABLE `pa_estimated_VO2` (
  `id`                                  VARCHAR(20) NOT NULL,
  `ts`                                  DATETIME        NOT NULL,
  `demographic_vo2_max`                 DECIMAL(6,2),
  `demographic_vo2_max_error`           DECIMAL(6,2),
  `filtered_demographic_vo2_max`        DECIMAL(6,2),
  `filtered_demographic_vo2_max_error`  DECIMAL(6,2),
  PRIMARY KEY (`id`,`ts`),
  CONSTRAINT `fk_pa_vo2_id`
      FOREIGN KEY (`id`) REFERENCES `participants`(`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* ========= 4. 运动 Session 记录 =========== */
DROP TABLE IF EXISTS `pa_reports`;
CREATE TABLE `pa_reports` (
  `session_id`                      BIGINT AUTO_INCREMENT PRIMARY KEY,
  `id`                              VARCHAR(20) NOT NULL,
  `record_ts`                       DATETIME    NOT NULL,      -- CSV 的 timestamp
  `activity_name`                   VARCHAR(50),
  `average_heart_rate`              DECIMAL(5,2),
  `duration`                    INT UNSIGNED,              -- duration
  `active_duration`             INT UNSIGNED,
  `calories`                   DECIMAL(8,2),
  `steps`                           INT UNSIGNED,

  `sedentary_minutes`               SMALLINT UNSIGNED,
  `_lightly_active_minutes`          SMALLINT UNSIGNED,
  `fairly_active_minutes`           SMALLINT UNSIGNED,
  `very_active_minutes`             SMALLINT UNSIGNED,

  `out_of_range_minutes`            SMALLINT UNSIGNED,
  `out_of_range_minimum_heart_rate`             DECIMAL(5,2),
  `out_of_range_maximum_heart_rate`             DECIMAL(5,2),
  `out_of_range_calories`           DECIMAL(8,2),

  `fat_burn_minutes`                SMALLINT UNSIGNED,
  `fat_burn_minimum_heart_rate`                 DECIMAL(5,2),
  `fat_burn_maximum_heart_rate`                 DECIMAL(5,2),
  `fat_burn_calories`               DECIMAL(8,2),

  `cardio_minutes`                  SMALLINT UNSIGNED,
  `cardio_minimum_heart_rate`                   DECIMAL(5,2),
  `cardio_maximum_heart_rate`                   DECIMAL(5,2),
  `cardio_calories`                 DECIMAL(8,2),

  `peak_minutes`                    SMALLINT UNSIGNED,
  `peak_minimum_heart_rate`                     DECIMAL(5,2),
  `peak_maximum_heart_rate`                     DECIMAL(5,2),
  `peak_calories`                   DECIMAL(8,2),

  INDEX `idx_pa_reports_id_ts` (`id`,`record_ts`),
  CONSTRAINT `fk_pa_reports_id`
      FOREIGN KEY (`id`) REFERENCES `participants`(`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* ---------------------------------------------------------
 * 7. DS8 – Sleep Activity
 * --------------------------------------------------------- */

DROP TABLE IF EXISTS `additional_sleep`;
CREATE TABLE `additional_sleep` (
  `sleep_id`          BIGINT      NOT NULL,
  `id`                VARCHAR(20) NOT NULL,
  `start_time`        DATETIME,
  `end_time`          DATETIME,
  `duration`      SMALLINT,
  `minutes_asleep`    SMALLINT,
  `minutes_awake`     SMALLINT,
  `minutes_in_bed`    SMALLINT,
  `main_sleep`        TINYINT(1),
  `minutes_in_deep_sleep`   SMALLINT,
  `minutes_in_light_sleep`  SMALLINT,
  `minutes_in_rem`    SMALLINT,
  PRIMARY KEY (`sleep_id`),
  INDEX (`id`),
  FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
     ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `oxygen_sat_minute`;
CREATE TABLE `oxygen_sat_minute` (
  `id`   VARCHAR(20) NOT NULL,
  `ts`   DATETIME    NOT NULL,
  `oxygen_saturation_by_minute` DECIMAL(5,2),
  PRIMARY KEY (`id`,`ts`),
  FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
    ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `oxygen_sat_daily`;
CREATE TABLE `oxygen_sat_daily` (
  `id`       VARCHAR(20) NOT NULL,
  `ts`     DATETIME        NOT NULL,
  `sleep_average_oxygen_saturation` DECIMAL(5,2),
  `lower_bound_oxygen_saturation` DECIMAL(5,2),
  `upper_bound_oxygen_saturation` DECIMAL(5,2),
  PRIMARY KEY (`id`,`ts`),
  FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
    ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

DROP TABLE IF EXISTS `skin_temp_wrist_minute`;
CREATE TABLE `skin_temp_wrist_minute` (
  `id`                    VARCHAR(20)  NOT NULL,        -- 参与者 ID
  `ts`                    DATETIME     NOT NULL,        -- 采样时间
  `temperature_difference`DECIMAL(5,2),                 -- 与基线的温度差 (°C)
  PRIMARY KEY (`id`,`ts`),
  CONSTRAINT `fk_wrist_temp_id`
      FOREIGN KEY (`id`) REFERENCES `participants`(`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

DROP TABLE IF EXISTS `skin_temp_sleep_nightly`;
CREATE TABLE `skin_temp_sleep_nightly` (
  `id`                                       VARCHAR(20)  NOT NULL,
  `start_sleep`                              DATETIME     NOT NULL,
  `end_sleep`                                DATETIME     NOT NULL,
  `temperature_samples`                      INT UNSIGNED,
  `nightly_temperature`                      DECIMAL(5,2),   -- °C
  `baseline_relative_sample_sum`             DECIMAL(10,4),
  `baseline_relative_sample_sum_of_squares`  DECIMAL(12,4),
  `baseline_relative_nightly_standard_deviation`   DECIMAL(6,4),
  `baseline_relative_sample_standard_deviation`    DECIMAL(6,4),
  PRIMARY KEY (`id`,`start_sleep`),               -- 一晚一行
  CONSTRAINT `fk_sleep_temp_id`
      FOREIGN KEY (`id`) REFERENCES `participants`(`id`)
      ON UPDATE CASCADE ON DELETE CASCADE,
  INDEX `idx_sleep_end` (`end_sleep`)             -- 便于按结束时间查询
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


DROP TABLE IF EXISTS `heart_rate_variability`;
CREATE TABLE `heart_rate_variability` (
  `id`       VARCHAR(20) NOT NULL,
  `ts`       DATETIME    NOT NULL,
  `rmssd` DECIMAL(6,2),
  `nrem_hr` DECIMAL(6,2),
  `entropy` DECIMAL(6,2),
  PRIMARY KEY (`id`,`ts`),
  FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
    ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* =========  Respiratory-rate (nightly summary)  ========= */
DROP TABLE IF EXISTS `respiratory_rate`;
CREATE TABLE `respiratory_rate` (
  `id`                                   VARCHAR(20)  NOT NULL,   -- 参与者 ID
  `night_end`                            DATETIME     NOT NULL,   -- CSV 中的 timestamp（睡眠段结束）
  
  /* ---- 整晚睡眠 ---- */
  `full_sleep_breathing_rate`            DECIMAL(4,1),
  `full_sleep_standard_deviation`        DECIMAL(4,1),
  `full_sleep_signal_to_noise`           DECIMAL(6,2),
  
  /* ---- 深睡 ---- */
  `deep_sleep_breathing_rate`            DECIMAL(4,1),
  `deep_sleep_standard_deviation`        DECIMAL(4,1),
  `deep_sleep_signal_to_noise`           DECIMAL(6,2),

  /* ---- 浅睡 ---- */
  `light_sleep_breathing_rate`           DECIMAL(4,1),
  `light_sleep_standard_deviation`       DECIMAL(4,1),
  `light_sleep_signal_to_noise`          DECIMAL(6,2),

  /* ---- REM 睡 ---- */
  `rem_sleep_breathing_rate`             DECIMAL(4,1),
  `rem_sleep_standard_deviation`         DECIMAL(4,1),
  `rem_sleep_signal_to_noise`            DECIMAL(6,2),
  
  PRIMARY KEY (`id`, `night_end`),
  CONSTRAINT `fk_resprate_id`
      FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


DROP TABLE IF EXISTS `sleep_quality`;
CREATE TABLE `sleep_quality` (
  `id`                 VARCHAR(20) NOT NULL,
  `start_time`         DATETIME    NOT NULL,
  `end_time`           DATETIME    NOT NULL,
  `overall_score`      TINYINT,
  `composition_score`  TINYINT,
  `revitalization_score` TINYINT,
  `duration_score`     TINYINT,
  `resting_heart_rate` TINYINT,
  `restlessness`       DECIMAL(8,7),
  PRIMARY KEY (`id`,`start_time`),
  CONSTRAINT `fk_sleep_quality_participant`
    FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
    ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


/* =========  Oviedo Sleep Questionnaire  ========= */
DROP TABLE IF EXISTS `OviedoSleepQuestionnaire`;
CREATE TABLE `OviedoSleepQuestionnaire` (
  `id`                                   VARCHAR(20)   NOT NULL,
  `visit`                                TINYINT UNSIGNED NOT NULL,

  /* ---------- 单题得分（1–7 级别；缺失允许 NULL） ---------- */
  `1_satisfaction_sleep`                 TINYINT,
  `2_1_initiate_sleep`                   TINYINT,
  `2_2_remain_asleep`                    TINYINT,
  `2_3_restorative_sleep`                TINYINT,
  `2_4_usual_waking_up`                  TINYINT,
  `2_5_excessive_somnolence`             TINYINT,
  `3_fall_asleep`                        TINYINT,
  `4_wake_up_night`                      TINYINT,
  `5_wake_up_earlier`                    TINYINT,

  /* ---------- 量化睡眠时长与效率 ---------- */
  `6a_hours_sleep`                       DECIMAL(4,2),   -- 每晚睡眠小时数
  `6b_hours_bed`                         DECIMAL(4,2),   -- 每晚卧床小时数
  `6c_sleep_efficiency`                  TINYINT,        -- 1–5 等级

  /* ---------- 白天主观症状 ---------- */
  `7_tiredness_not_sleep`                TINYINT,
  `8_somnolence_days`                    TINYINT,
  `9_somnolence_effects`                 TINYINT,

  /* ---------- 夜间相关症状 ---------- */
  `10a_snoring`                          TINYINT,
  `10b_snoring_suffocation`              TINYINT,
  `10c_leg_movements`                    TINYINT,
  `10d_nightmares`                       TINYINT,
  `10e_others`                           TINYINT,

  /* ---------- 其他 ---------- */
  `11_sleep_aids`                        TINYINT,

  /* ---------- 量表总分与维度分 ---------- */
  `sleep_satisfaction_score`             TINYINT,        -- 0–7
  `insomnia_score`                       SMALLINT UNSIGNED,  -- 0–36
  `somnolence_score`                     TINYINT,        -- 0–14
  `total_score`                          SMALLINT UNSIGNED,

  PRIMARY KEY (`id`, `visit`),
  CONSTRAINT `fk_oviedo_id`
      FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;



/* ---------------------------------------------------------
 * 8. DS9 – Emotional State
 * --------------------------------------------------------- */
/* =========  DASS-21（Depression Anxiety Stress Scales） ========= */
DROP TABLE IF EXISTS `emotional_dass21`;
CREATE TABLE `emotional_dass21` (
  `id`                      VARCHAR(20)      NOT NULL,
  `visit`                   TINYINT UNSIGNED NOT NULL,

  /* ---------- 21 个题项（建议 0–3 评分，TINYINT 足够） ---------- */
  `q1_wind_down`            TINYINT,
  `q2_mouth_dryness`        TINYINT,
  `q3_no_positive_feelings` TINYINT,
  `q4_difficulty_breathing` TINYINT,
  `q5_no_initiative`        TINYINT,
  `q6_overreact`            TINYINT,
  `q7_trembling`            TINYINT,
  `q8_nervous_energy`       TINYINT,
  `q9_panic`                TINYINT,
  `q10_no_prospects`        TINYINT,
  `q11_agitation`           TINYINT,
  `q12_no_relax`            TINYINT,
  `q13_downhearted`         TINYINT,
  `q14_intolerance`         TINYINT,
  `q15_close_to_panic`      TINYINT,
  `q16_no_enthusiasm`       TINYINT,
  `q17_selfworth`           TINYINT,
  `q18_touchy`              TINYINT,
  `q19_heart`               TINYINT,
  `q20_scared`              TINYINT,
  `q21_meaningless`         TINYINT,

  /* ---------- 3 个分量表分数 & 总分 ---------- */
  `depression_score`        SMALLINT UNSIGNED,   -- 0–42
  `stress_score`            SMALLINT UNSIGNED,   -- 0–42
  `anxiety_score`           SMALLINT UNSIGNED,   -- 0–42
  `total_score`             SMALLINT UNSIGNED,   -- 0–126

  PRIMARY KEY (`id`, `visit`),
  CONSTRAINT `fk_dass21_id`
      FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


/* ========== 1. 会话元信息表 ========== */
DROP TABLE IF EXISTS `eda_sessions`;
CREATE TABLE `eda_sessions` (
  `session_id`            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `id`                    VARCHAR(20)  NOT NULL,          -- 参与者 ID
  `session_ts`            DATETIME     NOT NULL,          -- 会话开始时间

  /* 心率 & HRV */
  `average_heart_rate`    DECIMAL(5,2),
  `start_heart_rate`      SMALLINT,
  `end_heart_rate`        SMALLINT,
  `hrv_baseline`          DECIMAL(6,2),

  /* 统计信息 */
  `sample_count`          INT UNSIGNED,                   -- 会话采样点数

  KEY `idx_eda_id_ts` (`id`, `session_ts`),

  CONSTRAINT `fk_eda_id`
      FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

/* ========== 2. 逐点皮电值长表 ========== */
DROP TABLE IF EXISTS `eda_levels`;
CREATE TABLE `eda_levels` (
  `session_id`            BIGINT UNSIGNED NOT NULL,
  `sample_idx`            INT UNSIGNED    NOT NULL,      -- 从 0 开始的采样序号
  `level_microsiemens`    DECIMAL(8,3),                  -- 皮电电导 (µS)

  PRIMARY KEY (`session_id`, `sample_idx`),

  CONSTRAINT `fk_eda_levels_session`
      FOREIGN KEY (`session_id`) REFERENCES `eda_sessions` (`session_id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


/* =========  Daily Stress Composite Score ========= */
DROP TABLE IF EXISTS `stress_daily_scores`;
CREATE TABLE `stress_daily_scores` (
  `id`                       VARCHAR(20)   NOT NULL,  -- 参与者 ID
  `ts`                       DATETIME          NOT NULL,  -- 由 timestamp 取日期

  `stress_score`             SMALLINT UNSIGNED,
  `sleep_points`             SMALLINT,
  `responsiveness_points`    SMALLINT,
  `exertion_points`          SMALLINT,

  PRIMARY KEY (`id`, `ts`),
  CONSTRAINT `fk_stress_id`
      FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

/* ---------------------------------------------------------
 * 9. DS10 – Additional Information (SUS)
 * --------------------------------------------------------- */
/* =========  System Usability Scale (SUS) ========= */
DROP TABLE IF EXISTS `sus_scores`;
CREATE TABLE `sus_scores` (
  `id`                     VARCHAR(20)      NOT NULL,         -- 参与者 ID
  `visit`                  TINYINT UNSIGNED NOT NULL,         -- 第几次随访

  /* ---------- 10 个题项 原样列名 ---------- */
  `1_like_to_use`         TINYINT,                           -- 1_like_to_use
  `2_complex`             TINYINT,                           -- 2_complex
  `3_easy_to_use`         TINYINT,                           -- 3_easy_to_use
  `4_technical_support`        TINYINT,                           -- 4_technical_support
  `5_well_integrated`     TINYINT,                           -- 5_well_integrated
  `6_inconsistency`        TINYINT,                           -- 6_inconsistency
  `7_quick_to_learn`      TINYINT,                           -- 7_quick_to_learn
  `8_cumbersome`          TINYINT,                           -- 8_cumbersome
  `9_confident`           TINYINT,                           -- 9_confident
  `10_need_to_learn`      TINYINT,                           -- 10_need_to_learn

  /* ---------- 计算列 ---------- */
  `positive_score`         SMALLINT,                          -- 正向项加总×2.5
  `negative_score`         SMALLINT,                          -- 反向项处理后×2.5
  `sum`                SMALLINT,                          -- CSV 列 “sum”
  `total_score`            SMALLINT,                          -- 0–100

  PRIMARY KEY (`id`, `visit`),
  CONSTRAINT `fk_sus_id`
      FOREIGN KEY (`id`) REFERENCES `participants` (`id`)
      ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;


/* ============  End of schema.sql  ============ */
