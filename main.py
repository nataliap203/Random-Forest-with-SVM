import os
from study_scripts.mushrooms_study import mushrooms_study
from study_scripts.wine_study import wine_study
from study_scripts.cancer_study import cancer_study
from study_scripts.crop_study import crop_study

RAPORTS_DIR_NAME = "raports"
os.makedirs(RAPORTS_DIR_NAME, exist_ok=True)


def main():
    mushrooms_study(iterations=25, raports_dir_name=RAPORTS_DIR_NAME)
    wine_study(iterations=25, raports_dir_name=RAPORTS_DIR_NAME)
    crop_study(iterations=25, raports_dir_name=RAPORTS_DIR_NAME)
    cancer_study(iterations=25, raports_dir_name=RAPORTS_DIR_NAME)


if __name__ == "__main__":
    main()
