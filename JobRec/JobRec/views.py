from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.urls import reverse
import requests
from bs4 import BeautifulSoup 
from .preprocess import *
from .recommend import *
import pandas as pd
# Document preprocessing steps
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

filename = 'retrieved_products.csv'

def home(request):
    print("Inside home")
    if request.method == "POST":
        df_jobs = pd.read_csv("Combined_Jobs_Final.csv")
        # print("1: ", df_jobs.head(1))
        cols = ['Job.ID']+['Title']+['Position']+ ['Company']+['City']+['Employment.Type']+['Job.Description']
        df_jobs = df_jobs[cols]
        df_jobs.columns = ['Job.ID', 'Title', 'Position', 'Company','City', 'Empl_type','Job_Description']
        # print("2: ", df_jobs.head(1)) 

        df_nan_city = df_jobs[pd.isnull(df_jobs['City'])]
        # print("3: ", df_nan_city.head(1))
        # print("4: ", df_nan_city.groupby(['Company'])['City'].count())

        #replacing nan with thier headquarters location
        df_jobs['Company'] = df_jobs['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')
        df_jobs.loc[df_jobs.Company == 'CHI Payment Systems', 'City'] = 'Illinois'
        df_jobs.loc[df_jobs.Company == 'Academic Year In America', 'City'] = 'Stamford'
        df_jobs.loc[df_jobs.Company == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'
        df_jobs.loc[df_jobs.Company == 'Driveline Retail', 'City'] = 'Coppell'
        df_jobs.loc[df_jobs.Company == 'Educational Testing Services', 'City'] = 'New Jersey'
        df_jobs.loc[df_jobs.Company == 'Genesis Health System', 'City'] = 'Davennport'
        df_jobs.loc[df_jobs.Company == 'Home Instead Senior Care', 'City'] = 'Nebraska'
        df_jobs.loc[df_jobs.Company == 'St. Francis Hospital', 'City'] = 'New York'
        df_jobs.loc[df_jobs.Company == 'Volvo Group', 'City'] = 'Washington'
        df_jobs.loc[df_jobs.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'

        # print("5: ", df_jobs.isnull().sum())
        df_nan_emp = df_jobs[pd.isnull(df_jobs['Empl_type'])]
        # print("6: ", df_nan_emp.head(1))
        # print("7: ", df_jobs['Empl_type'].unique())


        #replacing na values with part time/full time
        df_jobs['Empl_type']=df_jobs['Empl_type'].fillna('Full-Time/Part-Time')
        # print("8: ", df_jobs.groupby(['Empl_type'])['Company'].count())
        # print("9: ", df_jobs.head())

        df_jobs["text"] = df_jobs["Position"].map(str) + " " + df_jobs["Company"] +" "+ df_jobs["City"]+ " "+df_jobs['Empl_type']+" "+df_jobs['Job_Description'] +" "+df_jobs['Title']
        # print("10: ", df_jobs.head(2))
        df_all = df_jobs[['Job.ID', 'text', 'Title']]
        df_all = df_all.fillna(" ")
        # print("11: ", df_all.head(1))
        df_all = preprocessing(df_all)
        # print("12: ", df_all)
        # print("13: ", df_all["text"])


        #tf-idf
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_jobid = tfidf_vectorizer.fit_transform((df_all['text']))

        #Creating user corpus
        df_job_view = pd.read_csv("Job_Views.csv")
        df_job_view = df_job_view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]
        df_job_view["select_pos_com_city"] = df_job_view["Position"].map(str) + "  " + df_job_view["Company"] +"  "+ df_job_view["City"]
        df_job_view['select_pos_com_city'] = applying(df_job_view['select_pos_com_city'].map(str))
        # df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].map(str).apply(clean_txt)
        df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].str.lower()
        df_job_view = df_job_view[['Applicant.ID','select_pos_com_city']]

        #Experience dataset
        df_experience = pd.read_csv("Experience.csv")
        #taking only Position
        df_experience= df_experience[['Applicant.ID','Position.Name']]
        #cleaning the text
        df_experience['Position.Name'] = applying(df_experience['Position.Name'].map(str))
        # df_experience['Position.Name'] = df_experience['Position.Name'].map(str).apply(clean_txt)

        df_experience =  df_experience.sort_values(by='Applicant.ID')
        df_experience = df_experience.fillna(" ")

        df_experience = df_experience.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()

        #POI
        df_poi =  pd.read_csv("Positions_Of_Interest.csv", sep=',')
        df_poi = df_poi.sort_values(by='Applicant.ID')

        df_poi = df_poi.drop('Updated.At', 1)
        df_poi = df_poi.drop('Created.At', 1)

        #cleaning the text
        # print("14: ", df_poi['Position.Of.Interest'])
        df_poi['Position.Of.Interest'] = applying(df_poi['Position.Of.Interest'].map(str))
        # print("15: ", df_poi['Position.Of.Interest'])
        # df_poi['Position.Of.Interest']=df_poi['Position.Of.Interest'].map(str).apply(clean_txt)
        df_poi = df_poi.fillna(" ")

        df_poi = df_poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()

        #Creating the final user dataset by merging all the users datasets
        df_jobs_exp = df_job_view.merge(df_experience, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
        df_jobs_exp = df_jobs_exp.fillna(' ')
        df_jobs_exp = df_jobs_exp.sort_values(by='Applicant.ID')

        #Merging position of interest with existing dataframe
        df_jobs_exp_poi = df_jobs_exp.merge(df_poi, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
        df_jobs_exp_poi = df_jobs_exp_poi.fillna(' ')
        df_jobs_exp_poi = df_jobs_exp_poi.sort_values(by='Applicant.ID')

        #combining all the columns
        df_jobs_exp_poi["text"] = df_jobs_exp_poi["select_pos_com_city"].map(str) + df_jobs_exp_poi["Position.Name"] +" "+ df_jobs_exp_poi["Position.Of.Interest"]

        df_final_person= df_jobs_exp_poi[['Applicant.ID','text']]
        df_final_person.columns = ['Applicant_id','text']

        # print("16: ", df_final_person['text'])
        df_final_person['text'] = applying(df_final_person['text'])
        # print("17: ", df_final_person['text'])
        # df_final_person['text'] = df_final_person['text'].apply(clean_txt)

        index = np.where(df_final_person['Applicant_id'] == 326)[0][0]
        user_q = df_final_person.iloc[[index]]

        user_tfidf = tfidf_vectorizer.transform(user_q['text'])
        cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

        output2 = list(cos_similarity_tfidf)

        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
        list_scores = [output2[i][0][0] for i in top]
        rec = get_recommendation(top,df_all, list_scores)

        # print("Type: ", type(rec))
        # print("Rec: ", rec)
        # print("RecTen: ", rec[0:10])

        # for index, row in rec.iterrows():
        #     print(row['ApplicantID'])
            # print(row.ApplicantID, row.JobID, row.title, row.score)

        rows = []
        rows = [(index, row) for index, row in rec.iterrows()]
        # for index, row in rec.iterrows():
        #     rows.append((index, row))
        # print(rows)
        # for row in rows:
        #     print("RowHere", row)
        #     print("RowOver")
        #     print("----Row 0-----")
        #     print(row[0])
        #     print("----Over------")
        #     print("----Row 1-----")
        #     print(row[1]['ApplicantID'])
        #     print(row[1]['title'])
        #     print(row[1]['title'])
        #     print(row[1]['score'])
        #     print("---Over---")
            # print("2: ", row[2])
            # print("3: ", row[3])
        # return render(request, "display_jobs.html", {'rows': rec})
        return render(request, "display_jobs.html", {'rows': rows})

    return render(request,"home.html")


def count(request):
    if request.method == "POST":
        df_jobs = pd.read_csv("Combined_Jobs_Final.csv")
        # print("1: ", df_jobs.head(1))
        cols = ['Job.ID']+['Title']+['Position']+ ['Company']+['City']+['Employment.Type']+['Job.Description']
        df_jobs = df_jobs[cols]
        df_jobs.columns = ['Job.ID', 'Title', 'Position', 'Company','City', 'Empl_type','Job_Description']
        # print("2: ", df_jobs.head(1)) 

        df_nan_city = df_jobs[pd.isnull(df_jobs['City'])]
        # print("3: ", df_nan_city.head(1))
        # print("4: ", df_nan_city.groupby(['Company'])['City'].count())

        #replacing nan with thier headquarters location
        df_jobs['Company'] = df_jobs['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')
        df_jobs.loc[df_jobs.Company == 'CHI Payment Systems', 'City'] = 'Illinois'
        df_jobs.loc[df_jobs.Company == 'Academic Year In America', 'City'] = 'Stamford'
        df_jobs.loc[df_jobs.Company == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'
        df_jobs.loc[df_jobs.Company == 'Driveline Retail', 'City'] = 'Coppell'
        df_jobs.loc[df_jobs.Company == 'Educational Testing Services', 'City'] = 'New Jersey'
        df_jobs.loc[df_jobs.Company == 'Genesis Health System', 'City'] = 'Davennport'
        df_jobs.loc[df_jobs.Company == 'Home Instead Senior Care', 'City'] = 'Nebraska'
        df_jobs.loc[df_jobs.Company == 'St. Francis Hospital', 'City'] = 'New York'
        df_jobs.loc[df_jobs.Company == 'Volvo Group', 'City'] = 'Washington'
        df_jobs.loc[df_jobs.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'

        # print("5: ", df_jobs.isnull().sum())
        df_nan_emp = df_jobs[pd.isnull(df_jobs['Empl_type'])]
        # print("6: ", df_nan_emp.head(1))
        # print("7: ", df_jobs['Empl_type'].unique())


        #replacing na values with part time/full time
        df_jobs['Empl_type']=df_jobs['Empl_type'].fillna('Full-Time/Part-Time')
        # print("8: ", df_jobs.groupby(['Empl_type'])['Company'].count())
        # print("9: ", df_jobs.head())

        df_jobs["text"] = df_jobs["Position"].map(str) + " " + df_jobs["Company"] +" "+ df_jobs["City"]+ " "+df_jobs['Empl_type']+" "+df_jobs['Job_Description'] +" "+df_jobs['Title']
        # print("10: ", df_jobs.head(2))
        df_all = df_jobs[['Job.ID', 'text', 'Title']]
        df_all = df_all.fillna(" ")
        # print("11: ", df_all.head(1))
        df_all = preprocessing(df_all)
        # print("12: ", df_all)
        # print("13: ", df_all["text"])


        #tf-idf
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_jobid = tfidf_vectorizer.fit_transform((df_all['text']))

        #Creating user corpus
        df_job_view = pd.read_csv("Job_Views.csv")
        df_job_view = df_job_view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]
        df_job_view["select_pos_com_city"] = df_job_view["Position"].map(str) + "  " + df_job_view["Company"] +"  "+ df_job_view["City"]
        df_job_view['select_pos_com_city'] = applying(df_job_view['select_pos_com_city'].map(str))
        # df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].map(str).apply(clean_txt)
        df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].str.lower()
        df_job_view = df_job_view[['Applicant.ID','select_pos_com_city']]

        #Experience dataset
        df_experience = pd.read_csv("Experience.csv")
        #taking only Position
        df_experience= df_experience[['Applicant.ID','Position.Name']]
        #cleaning the text
        df_experience['Position.Name'] = applying(df_experience['Position.Name'].map(str))
        # df_experience['Position.Name'] = df_experience['Position.Name'].map(str).apply(clean_txt)

        df_experience =  df_experience.sort_values(by='Applicant.ID')
        df_experience = df_experience.fillna(" ")

        df_experience = df_experience.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()

        #POI
        df_poi =  pd.read_csv("Positions_Of_Interest.csv", sep=',')
        df_poi = df_poi.sort_values(by='Applicant.ID')

        df_poi = df_poi.drop('Updated.At', 1)
        df_poi = df_poi.drop('Created.At', 1)

        #cleaning the text
        # print("14: ", df_poi['Position.Of.Interest'])
        df_poi['Position.Of.Interest'] = applying(df_poi['Position.Of.Interest'].map(str))
        # print("15: ", df_poi['Position.Of.Interest'])
        # df_poi['Position.Of.Interest']=df_poi['Position.Of.Interest'].map(str).apply(clean_txt)
        df_poi = df_poi.fillna(" ")

        df_poi = df_poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()

        #Creating the final user dataset by merging all the users datasets
        df_jobs_exp = df_job_view.merge(df_experience, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
        df_jobs_exp = df_jobs_exp.fillna(' ')
        df_jobs_exp = df_jobs_exp.sort_values(by='Applicant.ID')

        #Merging position of interest with existing dataframe
        df_jobs_exp_poi = df_jobs_exp.merge(df_poi, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
        df_jobs_exp_poi = df_jobs_exp_poi.fillna(' ')
        df_jobs_exp_poi = df_jobs_exp_poi.sort_values(by='Applicant.ID')

        #combining all the columns
        df_jobs_exp_poi["text"] = df_jobs_exp_poi["select_pos_com_city"].map(str) + df_jobs_exp_poi["Position.Name"] +" "+ df_jobs_exp_poi["Position.Of.Interest"]

        df_final_person= df_jobs_exp_poi[['Applicant.ID','text']]
        df_final_person.columns = ['Applicant_id','text']

        # print("16: ", df_final_person['text'])
        df_final_person['text'] = applying(df_final_person['text'])
        # print("17: ", df_final_person['text'])
        # df_final_person['text'] = df_final_person['text'].apply(clean_txt)

        index = np.where(df_final_person['Applicant_id'] == 326)[0][0]
        user_q = df_final_person.iloc[[index]]

        user_tfidf = tfidf_vectorizer.transform(user_q['text'])
        cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

        output2 = list(cos_similarity_tfidf)

        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
        list_scores = [output2[i][0][0] for i in top]
        count_vectorizer = CountVectorizer()
        count_jobid = count_vectorizer.fit_transform((df_all['text']))
        print("count_jobid: ", count_jobid)
        user_count = count_vectorizer.transform(user_q['text'])
        cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_jobid)
        output2 = list(cos_similarity_countv)
        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
        list_scores = [output2[i][0][0] for i in top]
        rec = get_recommendation(top,df_all, list_scores)
        rows = []
        rows = [(index, row) for index, row in rec.iterrows()]
        return render(request, "display_jobs.html", {'rows': rows})
    return render(request,"home.html")

# def spacy(request):
#     return render(request,"output.html")

def knn(request):
    if request.method == "POST":
        df_jobs = pd.read_csv("Combined_Jobs_Final.csv")
        # print("1: ", df_jobs.head(1))
        cols = ['Job.ID']+['Title']+['Position']+ ['Company']+['City']+['Employment.Type']+['Job.Description']
        df_jobs = df_jobs[cols]
        df_jobs.columns = ['Job.ID', 'Title', 'Position', 'Company','City', 'Empl_type','Job_Description']
        # print("2: ", df_jobs.head(1)) 

        df_nan_city = df_jobs[pd.isnull(df_jobs['City'])]
        # print("3: ", df_nan_city.head(1))
        # print("4: ", df_nan_city.groupby(['Company'])['City'].count())

        #replacing nan with thier headquarters location
        df_jobs['Company'] = df_jobs['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')
        df_jobs.loc[df_jobs.Company == 'CHI Payment Systems', 'City'] = 'Illinois'
        df_jobs.loc[df_jobs.Company == 'Academic Year In America', 'City'] = 'Stamford'
        df_jobs.loc[df_jobs.Company == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'
        df_jobs.loc[df_jobs.Company == 'Driveline Retail', 'City'] = 'Coppell'
        df_jobs.loc[df_jobs.Company == 'Educational Testing Services', 'City'] = 'New Jersey'
        df_jobs.loc[df_jobs.Company == 'Genesis Health System', 'City'] = 'Davennport'
        df_jobs.loc[df_jobs.Company == 'Home Instead Senior Care', 'City'] = 'Nebraska'
        df_jobs.loc[df_jobs.Company == 'St. Francis Hospital', 'City'] = 'New York'
        df_jobs.loc[df_jobs.Company == 'Volvo Group', 'City'] = 'Washington'
        df_jobs.loc[df_jobs.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'

        # print("5: ", df_jobs.isnull().sum())
        df_nan_emp = df_jobs[pd.isnull(df_jobs['Empl_type'])]
        # print("6: ", df_nan_emp.head(1))
        # print("7: ", df_jobs['Empl_type'].unique())


        #replacing na values with part time/full time
        df_jobs['Empl_type']=df_jobs['Empl_type'].fillna('Full-Time/Part-Time')
        # print("8: ", df_jobs.groupby(['Empl_type'])['Company'].count())
        # print("9: ", df_jobs.head())

        df_jobs["text"] = df_jobs["Position"].map(str) + " " + df_jobs["Company"] +" "+ df_jobs["City"]+ " "+df_jobs['Empl_type']+" "+df_jobs['Job_Description'] +" "+df_jobs['Title']
        # print("10: ", df_jobs.head(2))
        df_all = df_jobs[['Job.ID', 'text', 'Title']]
        df_all = df_all.fillna(" ")
        # print("11: ", df_all.head(1))
        df_all = preprocessing(df_all)
        # print("12: ", df_all)
        # print("13: ", df_all["text"])


        #tf-idf
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_jobid = tfidf_vectorizer.fit_transform((df_all['text']))

        #Creating user corpus
        df_job_view = pd.read_csv("Job_Views.csv")
        df_job_view = df_job_view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]
        df_job_view["select_pos_com_city"] = df_job_view["Position"].map(str) + "  " + df_job_view["Company"] +"  "+ df_job_view["City"]
        df_job_view['select_pos_com_city'] = applying(df_job_view['select_pos_com_city'].map(str))
        # df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].map(str).apply(clean_txt)
        df_job_view['select_pos_com_city'] = df_job_view['select_pos_com_city'].str.lower()
        df_job_view = df_job_view[['Applicant.ID','select_pos_com_city']]

        #Experience dataset
        df_experience = pd.read_csv("Experience.csv")
        #taking only Position
        df_experience= df_experience[['Applicant.ID','Position.Name']]
        #cleaning the text
        df_experience['Position.Name'] = applying(df_experience['Position.Name'].map(str))
        # df_experience['Position.Name'] = df_experience['Position.Name'].map(str).apply(clean_txt)

        df_experience =  df_experience.sort_values(by='Applicant.ID')
        df_experience = df_experience.fillna(" ")

        df_experience = df_experience.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()

        #POI
        df_poi =  pd.read_csv("Positions_Of_Interest.csv", sep=',')
        df_poi = df_poi.sort_values(by='Applicant.ID')

        df_poi = df_poi.drop('Updated.At', 1)
        df_poi = df_poi.drop('Created.At', 1)

        #cleaning the text
        # print("14: ", df_poi['Position.Of.Interest'])
        df_poi['Position.Of.Interest'] = applying(df_poi['Position.Of.Interest'].map(str))
        # print("15: ", df_poi['Position.Of.Interest'])
        # df_poi['Position.Of.Interest']=df_poi['Position.Of.Interest'].map(str).apply(clean_txt)
        df_poi = df_poi.fillna(" ")

        df_poi = df_poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()

        #Creating the final user dataset by merging all the users datasets
        df_jobs_exp = df_job_view.merge(df_experience, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
        df_jobs_exp = df_jobs_exp.fillna(' ')
        df_jobs_exp = df_jobs_exp.sort_values(by='Applicant.ID')

        #Merging position of interest with existing dataframe
        df_jobs_exp_poi = df_jobs_exp.merge(df_poi, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
        df_jobs_exp_poi = df_jobs_exp_poi.fillna(' ')
        df_jobs_exp_poi = df_jobs_exp_poi.sort_values(by='Applicant.ID')

        #combining all the columns
        df_jobs_exp_poi["text"] = df_jobs_exp_poi["select_pos_com_city"].map(str) + df_jobs_exp_poi["Position.Name"] +" "+ df_jobs_exp_poi["Position.Of.Interest"]

        df_final_person= df_jobs_exp_poi[['Applicant.ID','text']]
        df_final_person.columns = ['Applicant_id','text']

        # print("16: ", df_final_person['text'])
        df_final_person['text'] = applying(df_final_person['text'])
        # print("17: ", df_final_person['text'])
        # df_final_person['text'] = df_final_person['text'].apply(clean_txt)

        index = np.where(df_final_person['Applicant_id'] == 326)[0][0]
        user_q = df_final_person.iloc[[index]]

        user_tfidf = tfidf_vectorizer.transform(user_q['text'])
        cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

        output2 = list(cos_similarity_tfidf)

        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
        list_scores = [output2[i][0][0] for i in top]
        count_vectorizer = CountVectorizer()
        count_jobid = count_vectorizer.fit_transform((df_all['text']))
        print("count_jobid: ", count_jobid)
        user_count = count_vectorizer.transform(user_q['text'])
        cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_jobid)
        output2 = list(cos_similarity_countv)
        top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
        list_scores = [output2[i][0][0] for i in top]
        n_neighbors = 11
        KNN = NearestNeighbors(n_neighbors=n_neighbors, p=2)
        KNN.fit(tfidf_jobid)
        NNs = KNN.kneighbors(user_tfidf, return_distance=True)
        print(NNs[0][0][1:])
        top = NNs[1][0][1:]
        index_score = NNs[0][0][1:]
        rec = get_recommendation(top, df_all, index_score)
        rows = []
        rows = [(index, row) for index, row in rec.iterrows()]
        return render(request, "display_jobs.html", {'rows': rows})
    return render(request,"home.html")


