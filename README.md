Dataset Overview
This project utilizes the Emergency Triage: Venomous Bites Dataset sourced from Kaggle. It is a comprehensive, synthetic dataset containing exactly one million clinical triage records of patients who have suffered from venomous bites or stings.

The core challenge presented by this data lies in its unique medical context. Every patient in the dataset is simulated as "non-verbal"—whether they are in severe medical shock, facing language barriers, or mute. Consequently, medical staff cannot ask the patients what bit them. Instead, they must rely entirely on objective clinical symptoms, such as heart rate, blood pressure, and other physical indicators, to correctly diagnose the venom source and administer the appropriate antivenom.

Key Features of the Data:

Volume: The dataset contains 1,000,000 rows, providing a robust foundation for training complex machine learning models without overloading standard hardware memory.

Target Classes: The data is perfectly balanced across four specific classification targets: Harmless_Insect, Viper_Snake, Scorpion, and Black_Widow_Spider.

Reference:

Ferretti, J. (n.d.). Emergency Triage: Venomous Bites Dataset. Kaggle. Available at: https://www.kaggle.com/datasets/jacopoferretti/emergency-triage-venomous-bites-dataset
