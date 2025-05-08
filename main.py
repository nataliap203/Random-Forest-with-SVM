import os
from study_scripts.mushrooms_study import mushrooms_grid
from study_scripts.wine_study import wine_grid
from study_scripts.cancer_study import cancer_grid

RAPORTS_DIR_NAME = "RAPORTS"
os.makedirs(RAPORTS_DIR_NAME, exist_ok=True)

def main():
    mushrooms_grid(RAPORTS_DIR_NAME)
    # wine_grid(RAPORTS_DIR_NAME)
    # cancer_grid(RAPORTS_DIR_NAME)


if __name__ == "__main__":
    main()