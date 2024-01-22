from pyperclip import paste

from response_generation import categorize_offer


def data_loading_interface():
    print("POSSIBLE COMMANDS:"
          "\n1. You can set/switch to a category of offers you're pasting in - write 'l' for low-income, 'h' for high income, 'p' for part-time, submit with ENTER"
          "\n2. Press just ENTER to paste copied text from your clipboard"
          "\n\ta) Paste url of the description"
          "\n\tb) Paste job description"
          "\n3. Write  'submit' and press ENTER to stop pasting and start generation of resumes")

    category = None
    url = None
    job_desc = None

    offers = []

    while True:
        command = input(f"\nSet category: {category}, set category/press ENTER to paste: ")

        if 'l' in command:
            category = "low_income"
            continue
        elif 'h' in command:
            category = "high_income"
            continue
        elif 'p' in command:
            category = "part_time"
            continue

        if category is None:
            print("You must first set a category of offers you're browsing "
                  "'l' for low-income, 'h' for high income, 'p' for part-time")
            continue

        text = paste()

        if command == "submit" or command == "done" or command == "DONE":
            submit_offers(offers)

        elif text.startswith("http"):
            url = text
            print("URL PASTED")
        elif url is None:
            print("Paste url first, then job description")
        else:
            job_desc = text
            print("DESCRIPTION PASTED: ", job_desc[:100], "...")

        if url is not None and job_desc is not None:
            offers.append({
                "url": url,
                "job_desc": job_desc,
                "category": category
            })

            url = None
            job_desc = None

            print("\n---------- OFFER SAVED FOR SUBMISSION ----------")


def submit_offers(offers):
    print("Submitting ", len(offers), " offers...")
    for offer in offers:
        try:
            categorize_offer(offer["url"], offer["category"], offer["job_desc"])
        except Exception as e:
            print(e)
            print("Something went wrong with offer at url: ", offer["url"])

    print("ALL OFFERS SUBMITTED")
    exit(0)

if __name__ == '__main__':
    data_loading_interface()
