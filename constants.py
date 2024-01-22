import os

ROMA_FIRST_NAMES = {
    "male": ["Laco", "Rudo", "Fero", "Maroš", "Martin", "Tomáš", "Ernesto", "Ernik", "Onur", "Patrik", "Mario"],
    "female": ["Esmeralda", "Jenifer", "Dominika", "Karol", "Šarlota", "Eržika", "Denisa", "Šerezáda"]
}

ROMA_LAST_NAMES = {
    "male": ["Lakatoš", "Horváth", "Ožana", "Gažlík", "Saržok", "Žiga", "Kandráč", "Terkeyl", "Ferko", "Tokár", "Tancoš"],
    "female": ["Lakatošová", "Horváthová", "Ožananová", "Gažlíková", "Saržoková", "Žigová", "Kandráčová", "Terkeylová", "Ferková", "Tokárová", "Tancošová"]
}

WHITE_FIRST_NAMES = {
    "male": ["Jiří", "Jan", "Petr", "Josef", "Pavel", "Martin", "Jaroslav", "Tomáš"],
    "female": ["Marie", "Jana", "Eva", "Hana", "Anna", "Lenka", "Kateřina", "Lucie"]
}

WHITE_LAST_NAMES = {
    "male": ["Novák", "Svoboda", "Novotný", "Dvořák", "Černý", "Kučera", "Veselý", "Krejčí", "Horák"],
    "female": ["Nováková", "Svobodová", "Novotná", "Dvořáková", "Černá", "Kučerová", "Veselá", "Horáková"]
}

# ranges on the number of experiences
HIGH_QUALIF_EXPS_RANGE = (4, 6)
LOW_QUALIF_EXPS_RANGE = (2, 3)

# emails for receiving responses from companies
ROMA_HIGH_QUALIF_EMAIL = "pracovniprihlaskyy+1@gmail.com"
ROMA_LOW_QUALIF_EMAIL = "pracovniprihlaskyy+2@gmail.com"
WHITE_HIGH_QUALIF_EMAIL = "pracovniprihlaskyy+3@gmail.com"
WHITE_LOW_QUALIF_EMAIL = "pracovniprihlaskyy+4@gmail.com"

OFFERS_EMAIL = "offerslowincome@gmail.com"
GCLOUD_EMAIL = "offersbrigady@gmail.com"

# Determining where should logs be saved
if os.getenv("RUN_LOCATION") == "LOCAL":
    CV_LOG_DIR = "generated-cvs"
elif os.getenv("RUN_LOCATION") == "DEPLOYED":
    CV_LOG_DIR = "/data/generated-cvs"