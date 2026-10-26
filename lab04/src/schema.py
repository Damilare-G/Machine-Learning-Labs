"""Schema, ranges and allowed categories for UCI Adult (the 'data dictionary')."""

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship",
               "race", "sex", "native-country"]
FEATURES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
            "native-country"]                       # the 14 input features
TARGET = "income"
ALL_COLUMNS = FEATURES + [TARGET]                   # 15 columns

EXPECTED_TYPES = {
    "age": "int", "workclass": "category", "fnlwgt": "int", "education": "category",
    "education-num": "int", "marital-status": "category", "occupation": "category",
    "relationship": "category", "race": "category", "sex": "category",
    "capital-gain": "int", "capital-loss": "int", "hours-per-week": "int",
    "native-country": "category", "income": "category",
}

RANGES = {"age": (17, 90), "hours-per-week": (1, 99), "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
MISSING_TOKEN = "?"

EDU_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, "11th": 7,
           "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
           "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}
MARRIED = ["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"]

ALLOWED = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                  "Without-pay", "Never-worked", "?"],
    "education": list(EDU_NUM),
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces", "?"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba",
                       "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
                       "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
                       "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
                       "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands",
                       "?"],
    "income": ["<=50K", ">50K"],
}
