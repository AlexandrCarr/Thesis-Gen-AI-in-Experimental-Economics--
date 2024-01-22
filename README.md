1. **Gmail API** — the website Jobs.cz periodacally sends a set of offers posted on that day to a central gmail inbox. Use use the Google Cloud service Gmail API to connect to that inbox and periodically retrieve all emails marked as unread. Upon succesfully reading the email it’s marked as read. The raw email’s body with styling is decoded from base64 format. Using regex parsing we find all URLs pointing to a job offer.
2. **Classification** — of offers — on the jobs.cz portal we set separate email notififications for low paying jobs, part time jobs and highly paid jobs using their own filtering system. Threshold for highly paid job is 50000 czk. The job type is noted in the email notification. If we have low paying offer we further classify it into a blue collar position and white collar position using GPT-3 API. If the offer is classified as white collar we use another GPT prompt to classify it as client facing or internally facing. Thus there are 5 possible classes in total.
3. **Generating resumes** — for each offer we sent a set of 4 resumes. These are combinations of white/romani names and low/high qualification. Qualification differs in the number of job experiences listed in the final resume. We randomly assign 4 different html CV templates. For each using GPT we generate ceratain number of job experiences. We prompt GPT to make the experiences relevant to the job description. We also choose random combination of white/romani name and surname. Thirdly an education experience is added to the resume. Similarly written by GPT.
4. **Submitting** — we use the selenium python library to automate interaction with the web form. We fill in the chosen name and also email at which we collect the responses. We find the HTML elements we want to click on by finding their id in the form’s HTML. 
5. **Hosting** — the python service can run on Render where a function executing the above steps is periodically called. The generated pdfs along with metadata are saved to the server’s persistent disk from where they can be later collected and used for data analysis.