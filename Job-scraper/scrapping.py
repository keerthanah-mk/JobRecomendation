import random
import time
import pandas as pd
from parsel import Selector
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
opts = Options()

driver = webdriver.Chrome(options=opts, executable_path="chromedriver")

#function to ensure all key data fields have a value
def validate_field(field):
    #if field is present pass
    if field:
        pass
    #if field is not present print text
    else:
        field = 'No results'
    return field

#driver.get() - > navigate to the page given by the URL address
driver.get('https://www.linkedin.com')

#locate email form by class_name
username = driver.find_element(By.ID, 'session_key')

#send keys to simulate key strokes
username.send_keys('keerthanahmurugesan@gmail.com')

#sleep for 0.5 seconds
sleep(0.5)

#locate password form by class_name
password = driver.find_element(By.ID, 'session_password')

#send keys to simulate key strokes
password.send_keys('ourFascinoLove')

#locate submit button by xpath
sign_in_button = driver.find_element(By.XPATH, '//*[@type="submit"]')

# .click() -> to mimic button click
sign_in_button.click()
sleep(15)


Jobdata = []
lnks = []
for x in range(0,20,10):
    driver.get(f'https://www.google.com/search?q=site%3Alinkedin.com%2Fin%2F+AND+%22Python+Developer%22+AND+%22Delhi%22&oq=site%3Alinkedin.com%2Fin%2F+AND+%22Python+Developer%22+AND+%22Delhi%22&aqs=chrome..69i57j69i58.4399j0j7&sourceid=chrome&ie=UTF-8')
    time.sleep(random.uniform(2.5, 4.9))
    linkedin_urls = [my_elem.get_attribute("href") for my_elem in WebDriverWait(driver, 20).until(EC.visibility_of_all_elements_located((By.XPATH, "//div[@class='yuRUbf']/a[@href]")))]
    lnks.append(linkedin_urls)

for x in lnks:
    for i in x:

        #get the profile URL
        driver.get(i)
        time.sleep(random.uniform(2.5, 4.9))

        #assigning the source code for the web page to variable sel
        sel = Selector(text=driver.page_source)

        # xpath to extract the text from the class containing the name
        name = sel.xpath('//*[starts-with(@class, "text-heading-xlarge inline t-24 v-align-middle break-words")]/text()').extract_first()

        #if name exists
        if name:
            # .strip() -> removes the new line and white spaces
            name = name.strip()

        # xpath to extract the text from the class containing the job title
        job_title = sel.xpath('//*[starts-with(@class, "text-body-medium break-words")]/text()').extract_first()

        if job_title:
            job_title = job_title.strip()

        try:
            #xpath to extract the text from the class containing the company
            company = driver.find_element(By.XPATH, '//ul[@class="pv-text-details__right-panel"]').text
        
        except:
            company = 'None'

        #xpath to extract the text from the class containing the location
        location = sel.xpath('//*[starts-with(@class, "text-body-small inline t-black--light break-words")]/text()').extract_first()

        if location:
            location = location.strip()

        #validating if the fields exist on the profile
        name = validate_field(name)
        job_title = validate_field(job_title)
        company = validate_field(company)
        location = validate_field(location)

        #printing the output
        print('\n')
        print('Name: ' + name)
        print('Job Title: ' + job_title)
        print('Company: ' + company)
        print('Location: ' + location)
        print('\n')

        data = {
                'Name' : name,
                'Job Title' : job_title,
                'Company' : company,
                'Location' : location
                }
        
        Jobdata.append(data)

df = pd.DataFrame(Jobdata)
df.to_csv('Jobdata_linkedin.csv', index=False)

#terminates the application
driver.quit()

print(Jobdata)

