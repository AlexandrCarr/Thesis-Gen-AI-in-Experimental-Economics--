import datetime
import re

from gpt_calls import *
from constants import *
from helpers import *

import random
import time
import requests
from bs4 import BeautifulSoup
import json
import pdfkit
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions
from datetime import datetime


def respond(url, category):
    response = requests.get(url)

    # We cannot send cvs through websites of the employer, only through the standardised jobs.cz one
    if "www.jobs.cz" not in response.url:
        print("offer is on the website of employer, not jobs.cz")
        return

    soup = BeautifulSoup(response.content, "html.parser")
    web_text = soup.get_text()

    start_search_i = web_text.find("Pracovní nabídka")

    print(web_text)

    if start_search_i == -1:
        print("error, the description of a job is not on the website")
        return

    job_desc = re.sub(r'\s{2,}', ' ', web_text[start_search_i + 16: start_search_i + 1000])

    categorize_offer(url, category, job_desc)


def categorize_offer(url, category, job_desc):
    if category == "low_income":
        # If it's a white or blue collar position
        blue_or_white_collar = classify_blue_white_collar(job_desc)
        category += "_" + blue_or_white_collar

        # If white collar position further classify if it's in contact with customers
        if blue_or_white_collar == "white":
            customer_or_internal_facing = classify_is_client_facing(job_desc)
            category += "_" + customer_or_internal_facing

    dir_path = os.path.join(CV_LOG_DIR, str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
    os.mkdir(dir_path)

    details = generate_application_details_and_pdfs(category, job_desc, url, dir_path)
    time.sleep(1)
    # submit_pdfs(url, dir_path, details)


def submit_pdfs(url, dir_path, details):
    name, surname = details["roma_hq_name"].split(" ")
    submit_pdf(url, name, surname, ROMA_HIGH_QUALIF_EMAIL, os.path.join(dir_path, "roma_hq.pdf"))

    name, surname = details["roma_lq_name"].split(" ")
    submit_pdf(url, name, surname, ROMA_LOW_QUALIF_EMAIL, os.path.join(dir_path, "roma_lq.pdf"))

    name, surname = details["white_hq_name"].split(" ")
    submit_pdf(url, name, surname, WHITE_HIGH_QUALIF_EMAIL, os.path.join(dir_path, "white_hq.pdf"))

    name, surname = details["white_lq_name"].split(" ")
    submit_pdf(url, name, surname, WHITE_LOW_QUALIF_EMAIL, os.path.join(dir_path, "white_lq.pdf"))


def submit_pdf(url, name, surname, email, cv_path):
    delay = random.uniform(2.0, 4.0)
    print("Sleep " + str(delay) + "s")
    time.sleep(delay)

    options = ChromeOptions()
    options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)



        answer_button = driver.find_element(By.XPATH,
                                     "//a[@class='Button Button--primary Button--large d-none d-tablet-inline-flex mr-tablet-700']")
        answer_button.click()

        name_field = driver.find_element(By.ID, "jobad_application_firstName")
        surname_field = driver.find_element(By.ID, "jobad_application_surname")
        email_field = driver.find_element(By.ID, "jobad_application_email")
        cv_file = driver.find_element(By.ID, "customCvs")

        name_field.clear()
        surname_field.clear()
        email_field.clear()

        name_field.send_keys(name)
        random.uniform(1.0, 2.0)
        surname_field.send_keys(surname)
        random.uniform(1.0, 2.0)
        email_field.send_keys(email)
        random.uniform(1.0, 2.0)
        cv_file.send_keys(os.path.abspath(cv_path))
        random.uniform(1.0, 2.0)

        submit_button = driver.find_element(By.XPATH,
                                     "//button[@class='Button Button--primary Button--large']")

        submit_button.click() # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! UNCOMMENT TO ACTUALLY SUBMIT THE CVs

        print("Form successfully submitted")
        driver.quit()

    except Exception as e:
        print("could not submit to the url, error: ", e)


def generate_application_details_and_pdfs(category, job_desc, url, dir_path):
    sex = random.choice(["male", "female"])
    hq_exps = random.randint(*HIGH_QUALIF_EXPS_RANGE)
    lq_exps = random.randint(*LOW_QUALIF_EXPS_RANGE)

    first_name_i = generate_unique_index((0, len(ROMA_FIRST_NAMES[sex]) - 1), [])
    last_name_i = generate_unique_index((0, len(ROMA_LAST_NAMES[sex]) - 1), [])
    roma_hq_name = ROMA_FIRST_NAMES[sex][first_name_i] + " " + ROMA_LAST_NAMES[sex][last_name_i]

    first_name_i = generate_unique_index((0, len(ROMA_FIRST_NAMES[sex]) - 1), [first_name_i])
    last_name_i = generate_unique_index((0, len(ROMA_LAST_NAMES[sex]) - 1), [last_name_i])
    roma_lq_name = ROMA_FIRST_NAMES[sex][first_name_i] + " " + ROMA_LAST_NAMES[sex][last_name_i]

    first_name_i = generate_unique_index((0, len(WHITE_FIRST_NAMES[sex]) - 1), [])
    last_name_i = generate_unique_index((0, len(WHITE_LAST_NAMES[sex]) - 1), [])
    white_hq_name = WHITE_FIRST_NAMES[sex][first_name_i] + " " + WHITE_LAST_NAMES[sex][last_name_i]

    first_name_i = generate_unique_index((0, len(WHITE_FIRST_NAMES[sex]) - 1), [first_name_i])
    last_name_i = generate_unique_index((0, len(WHITE_LAST_NAMES[sex]) - 1), [last_name_i])
    white_lq_name = WHITE_FIRST_NAMES[sex][first_name_i] + " " + WHITE_LAST_NAMES[sex][last_name_i]

    if category == "high_income":
        education_type = "vysoké škole"
    else:
        education_type = "střední škole"

    template_nums = [1, 2, 3, 4]
    random.shuffle(template_nums)

    details = {
        "category": category,
        "url" : url,
        "job_description": job_desc,
        "education_type": education_type,
        "sex": sex,
        "number_of_hq_exps": hq_exps,
        "number_of_lq_exps": lq_exps,
        "roma_hq_template" : template_nums[0],
        "roma_lq_template" : template_nums[1],
        "white_hq_template": template_nums[2],
        "white_lq_template": template_nums[3],
        "roma_hq_name": roma_hq_name,
        "roma_lq_name": roma_lq_name,
        "white_hq_name": white_hq_name,
        "white_lq_name": white_lq_name,
        "datetime": str(datetime.now())
    }

    generate_pdf(roma_hq_name.split(" ")[1] + roma_hq_name.split(" ")[0], roma_hq_name, hq_exps, template_nums[0], dir_path, details)
    generate_pdf(roma_lq_name.split(" ")[1] + " " + roma_lq_name.split(" ")[0], roma_lq_name, lq_exps, template_nums[1], dir_path, details)
    generate_pdf(white_hq_name, white_hq_name, hq_exps, template_nums[2], dir_path, details)
    generate_pdf(white_lq_name.lower(), white_lq_name, lq_exps, template_nums[3], dir_path, details)

    return details


def generate_pdf(tag, name, exps_num, template_number, dir_path, details):
    with open(f'cv-html-templates/template{template_number}.html', 'r') as file:
        html_content = file.read()

        job_exps = generate_job_experiences(details["job_description"], exps_num, details["sex"])
        education = generate_education(details["job_description"], details["education_type"], details["sex"])

        filled_html = html_content % {"name": name, "education": education, "job_exps": job_exps}

        # Saving file with metadata
        with open(os.path.join(dir_path, "details.json"), "w", encoding='utf8') as json_file:
            json.dump(details, json_file, indent=4, ensure_ascii=False)

        # generating the pdf
        pdfkit.from_string(filled_html, f"{dir_path}/{tag}.pdf")