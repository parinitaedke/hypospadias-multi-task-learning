# hypospadias-multi-task-learning

This project is focused on developing a multi-task-learning model to determine the HOPE and GMS scores for male infants with hypospadias using natural images.

This research is done as part of the Goldenberg Lab and SickKids Hospital.


## Quick Setup

#### Go to project directory
```
# Go to `hypospadias-multi-task-learning` repo directory
cd PATH/TO/DIRECTORY
```

#### 0. (Optional) Create virtual environmeent
```
# Create environment
python -m venv hypospadias_mtl

# Activate environment
# 1. In Windows
hypospadias_mtl\Scripts\activate
# 2. In Linux
source hypospadias_mtl/bin/activate
```

#### 1. Install dependencies
```
pip install -r requirements.txt
```

#### 2. Create symbolic link to data directory
```
# (In a console with admin priviliges)
# 1. In Windows
mklink /d ".\src\data" "PATH\TO\DATA\DIRECTORY"

# 2. In Linux
ln -s /PATH/TO/DATA/DIRECTORY ./src/data
```

#### 3. Modify model training specifications in main.py
```
# Update model training specifications in main.py as needed to run experiments.
```
