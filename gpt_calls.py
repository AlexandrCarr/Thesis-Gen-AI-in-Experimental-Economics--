import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def classify_blue_white_collar(job_desc):
    blue_or_white_collar_prompt = """
        Definition of a blue-collar worker: blue-collar workers are typically involved in manual or labor-intensive tasks that require physical effort and skill. They are often employed in industries such as manufacturing, construction, agriculture, mining, transportation, and maintenance. The work of blue-collar employees may involve operating machinery, assembling products, performing manual labor, and working in environments that might be physically demanding or hazardous. 
        Definition of a white-collar worker: White-collar workers are typically associated with professional, administrative, managerial, and office-based roles. They often work in sectors such as finance, information technology, healthcare, education, marketing, and various corporate settings. White-collar jobs tend to involve tasks that are more knowledge-based, involve decision-making, and may require skills such as communication, analysis, problem-solving, and coordination

        If the job description written in Czech langauge below describes a blue collar position ouput "blue" and if white collar position ouput "white".

        job description: \"\"\"
            {}
        \"\"\"
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=blue_or_white_collar_prompt.format(job_desc),
        max_tokens=1024,
        temperature=0.5
    ).choices[0].text.strip().lower()

    if "bl" in response: return "blue"
    elif "wh" in response: return "white"


def classify_is_client_facing(job_desc):
    is_client_facing_prompt = """
        Is the job description written in Czech langauge below describes a position that is customer-facing? A customer-facing position is one where you work directly with customers, helping them with their needs, questions, or concerns. If it is a customer-facing position output "customer-facing", and output "internal-facing" if it's not a customer-facing position.

        job description: \"\"\"
            {}
        \"\"\"
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=is_client_facing_prompt.format(job_desc),
        max_tokens=1024,
        temperature=0.5
    ).choices[0].text.strip().lower()

    if "cust" in response:
        return "customer-facing"
    elif "int" in response:
        return "internal-facing"


def generate_job_experiences(job_description, number_of_experiences, sex):
    czech_sex = "žena" if sex == "female" else "male"

    experience_prompt = """
    Přečti následující úryvek z pracovní nabídky:
    {}


    Vygeneruj {} pracovní zkušenosti ze stejného odvětní jako je výše uvedená praocovní nabídka. Použij reálné názvy firem a firmy musí být v češtině. Firmy Vynechej s.r.o. Názvy firem musí být různorodé a nesmí být to samé jak firma, do které se hlasí. Text bude následně použit v životopisu. Popis zkušenosti bude v bodech. Pohlaví vlastníka životopisu je {}. Nečísluj vygenerované pracovní zkušenosti. Datumy jednotlivých pracovních zkušeností se nesmí za žádnou cenu vzájemně překrývat - toto je velmi důležité! Datum první pracovní zkušenosti a všech následujících pracovních zkušeností musí začínat po datu ukončení vzdělání - toto je velmi důležité! Datum začátku první pracovní zkušenosti musí být po ukončení vzdělání. Datumy vzdělání a pracovních zkušeností se nesmí vzájemně překrývat. Je velmi důležité, aby datum ukončení vzdělání předcházelo datu zahájení první pracovní zkušenosti. Například pokud je vzdělání ukončeno v roce 2016, první pracovní zkušenost musí začít až v roce 2016, ne dříve. Taky se ujisti že názvy firem začínají velkým písmenem. Ujisti se že vygenerované nabídky odpovídají požadované kvalifikaci. Nikdy nepis veci typu objednal/a - vzdy pouze pis v tom rodu, ve kterem zivotopis ma byt. Jestli je to zivotopis muze, tak pis vzdy pouze v muzskem rode. Jestli je to zivotopis zeny, tak pis vzdy pouze v zenske rode. Dohlizej na to, ze vsechny vety jsou dopsany a ze nedelas zadne chyby. 

    Formát jedné prcovní zkušenosti:
    <h3><název firmy></h3>
    <b><datum začátku a datum konce></b>
    <p><popis náplňě práce></p>
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=experience_prompt.format(job_description, str(number_of_experiences), czech_sex),
        max_tokens=1024,
        temperature=0.9
    ).choices[0].text

    return response


def generate_education(job_description, education_type, sex):
    czech_sex = "žena" if sex == "female" else "male"

    experience_prompt = """
    Přečti následující úryvek z pracovní nabídky:
    {}
    
    Vygeneruj záznam o vzdělání dosaženém na {} z podobného oboru jako je výše uvedená pracovní nabídka. Použij reálné názvy škol. Text bude následně použit v životopisu. Pohlaví vlastníka životopisu je {}. 

    Formát jedné prcovní zkušenosti:
    <h3><název školy></h3>
    <b><datum začátku a datum konce></b>
    <p><obor školy></p>
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=experience_prompt.format(job_description, education_type, czech_sex),
        max_tokens=1024,
        temperature=0.7
    ).choices[0].text

    return response
