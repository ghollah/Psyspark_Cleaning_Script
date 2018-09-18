import requests
from flask import Flask, request
from flask import jsonify
import string
import json
import os
from pyspark.sql.functions import udf

url = "http://178.128.216.52:5000/etl/get_dataset"

payload = "\n\t\t{\n\t\t\"job_id\":\"35\", \n\t\t\"data_type\":\"file\"}\n\t\t\n"
headers = {
   'Content-Type': "application/json",
   'Cache-Control': "no-cache",
   'Postman-Token': "4de0555e-7403-48e8-b173-ff9c8de29b6a"
   }

response = requests.request("POST", url, data=payload, headers=headers)
data = response.json()
result = data['data']

#print(result)

import json 
import pandas as pd
import pyspark 
import findspark
findspark.init()
from pyspark.sql import *
from pyspark import *
import re

js = [json.dumps(result)]

#Importing mysql-connector with Spark Conf  
SUBMIT_ARGS = "--packages mysql:mysql-connector-java:5.1.39 pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS
conf = SparkConf()
sc = SparkContext(conf=conf).getOrCreate()
spark = SQLContext(sc)
jsonRDD = sc.parallelize(js)
df = spark.read.json(jsonRDD)


#Create a table to store the columns
#table = df.registerTempTable("MyTable")
#query = spark.sql("Select Country from MyTable").show(10)
#print(query)

app = Flask(__name__)

##To convert into lowercase
@app.route("/lower")
def lowerCase():
    
    #Fetching the data from mysql as a Spark dataframe
    #dataframe_mysql = spark.read.options(
   # dbtable = query).load()
    
    
    
    #importing UDF and String Type
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    #defining a udf to convert to lower case
    def low(x):
        try:
          return x.lower()
        except Exception as e:
          print("The error caused is ", e)
          

    #Making the lower UDF
    lowerUdf = udf(lambda y : low(y), StringType())

    #converting the dataframe into lowercase
    lower_df = df.select(lowerUdf("Country"))
    #print(lower_df)
    
    lower_df.write.format('jdbc').options(
    url='jdbc:mysql://127.0.0.1:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='lower',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    lower_df1 = lower_df.toJSON().collect()
        
    #returning the json
    return json.dumps(lower_df1)

##To carry out word replacement
@app.route("/replace")
def ReplaceFunc():
    #imports
    from pyspark.sql.functions import regexp_replace

    #Carrying out the replacement
    newDf = df.withColumn("Country", regexp_replace("Country", "Rwanda", "Kenya"))

    #saving the dataframe to a mysql table
    newDf.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='replaced',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    newDf1 = newDf.toJSON().collect()

    #returning the json
    return json.dumps(newDf1)
    

##To carry out normalization
@app.route("/normalize")
def NormalizeFunc():
    #accepting the column name to be fetched from database
    #"Country" = request.args.get(""Country"", None)

    #REMOVING PUNCTUATION
    from string import punctuation
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    #defining a function to remove punctuation    
    def Remove_punct(s):
      try:
        return ''.join(c for c in s if c not in punctuation)
      except Exception as e:
        print("The error caused is ", e)
        
        
    #Making the punctuation remove UDF
    Punct_RemoveUdf = udf(lambda y : Remove_punct(y), StringType())

    #removing the punctuations from the dataframe
    Punct_Remove_df = df.select(Punct_RemoveUdf("Country").alias("Country"))


    #Storing result of punctuation remove in the database
    Punct_Remove_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='PuncRemove',
    user='root',
    password='root').mode('append').save()



    #REMOVING NON_ASCII
    #defining a function to remove punctuation    
    def Remove_nonAscii(s):
      try:
        return ''.join([i if ord(i) < 128 else '' for i in s])
      except Exception as e:
        print("The error caused is ", e)
        

    #Making the Non Ascii remove UDF
    NonAscii_RemoveUdf = udf(lambda y : Remove_nonAscii(y), StringType())

    #removing the Non Ascii from the dataframe
    NonAscii_Remove_df = Punct_Remove_df.select(NonAscii_RemoveUdf("Country").alias("Country"))

    #Storing result Non Ascii removal in the database
    NonAscii_Remove_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='NonAsciiRemove',
    user='root',
    password='root').mode('append').save()



    #REPLACE NUMBERS
    import re

    #defining a function to replace numbers    
    def rep_Number(x):
      try:
        return re.sub("\d+", "", x)
      except Exception as e:
        print("The error caused is ", e)
        
        
    #Making the replace number UDF
    replaceNumUdf = udf(lambda y : rep_Number(y), StringType())

    #replacing the numbers from the dataframe
    replaceNum_df = NonAscii_Remove_df.select(replaceNumUdf("Country").alias("Country"))


    #Storing result of stemming in the database
    replaceNum_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='ReplaceNumber',
    user='root',
    password='root').mode('append').save()



    #imports
    from pyspark.ml.feature import Tokenizer
    from pyspark.ml.feature import StopWordsRemover

    #TOKENIZATION
    tokenizer = Tokenizer(inputCol="Country", outputCol="Tokens")
    token_df = tokenizer.transform(replaceNum_df).select("Country", "Tokens")

    #Storing result of tokenization in the database
    from pyspark.sql.functions import concat_ws, col
    token_df1 = token_df.withColumn('Tokens', concat_ws(',', 'Tokens'))

    token_df1.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='tokenized',
    user='root',
    password='root').mode('append').save()


    #STOP WORDS REMOVAL
    remover = StopWordsRemover()
    stopwords = remover.getStopWords()
    remover.setInputCol("Tokens")
    remover.setOutputCol("No_stop_words")
    vector_no_stopw_df = remover.transform(token_df).select("Country", "No_stop_words")

    #Storing result of stop words removal in the database
    vector_no_stopw_df1 = vector_no_stopw_df.withColumn('No_stop_words', concat_ws(' ', 'No_stop_words'))


    vector_no_stopw_df1.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='NoStopWords',
    user='root',
    password='root').mode('append').save()


    #STEMMING
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    #defining a function for stemming
    def stem(in_vec):
        try:
          out_vec = list()
          for t in in_vec:
            t_stem = stemmer.stem(t)
            if len(t_stem) > 2:
              out_vec.append(t_stem)
        except Exception as e:
          print("The error caused is ", e)
        return out_vec

    #Making the stemming UDF
    from pyspark.sql.types import ArrayType, StringType
    stemmer_udf = udf(lambda x: stem(x), ArrayType(StringType()))

    #Stemming the dataframe
    vector_stemmed_df = vector_no_stopw_df.withColumn("vector_stemmed", stemmer_udf("No_stop_words")).select("Country", "vector_stemmed")

    vector_stemmed_df1 = vector_stemmed_df.select("Country", col("vector_stemmed").alias("Stems"))

    #Storing result of stemming in the database
    vector_stemmed_df2 = vector_stemmed_df1.withColumn('Stems', concat_ws(',', 'Stems'))

    vector_stemmed_df2.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='Stemmed',
    user='root',
    password='root').mode('append').save()




    #LEMMATIZATION
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    #defining a function for lemmatizing
    def lem(in_vec):
        try:
          out_vec = []
          for t in in_vec:
            t_lem = lemmatizer.lemmatize(t)
            if len(t_lem) > 2:
              out_vec.append(t_lem)
        except Exception as e:
          print("The error caused is ", e)
        return out_vec

    #Making the lemmatizing UDF
    lemmer_udf = udf(lambda x: lem(x), ArrayType(StringType()))

    #Lemmatizing the dataframe
    vector_lem_df = vector_stemmed_df1.withColumn("vector_lemmed", lemmer_udf("Stems")).select("Country", "vector_lemmed")

    vector_lemmed_df1 = vector_lem_df.select("Country", col("vector_lemmed").alias("Lemmas"))


    #Storing result of lemmatizing in the database
    vector_lemmed_df2 = vector_lemmed_df1.withColumn('Lemmas', concat_ws(',', 'Lemmas'))

    vector_lemmed_df2.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='Lemmas',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    vector_lemmed_df3 = vector_lemmed_df2.toJSON().collect()

    #returning the json
    return json.dumps(vector_lemmed_df3)



##To extract emails from text
@app.route("/email")
def EmailExtract():
    
    #importing UDF and re
    from pyspark.sql.functions import udf
    import re
    from pyspark.sql.types import StringType

    #defining a udf to extract emails
    def email(x):
        return re.findall('\S+@\S+', x)

    #Making the Email UDF
    email_UDF = udf(lambda x: email(x), StringType())

    #Extracting emails from the dataframe
    email_df = df.withColumn("Country", email_UDF("Country"))

    #Converting list of emails into a string
    #email_df1 = email_df.withColumn('Email', concat_ws(',', 'Email'))

    #saving the dataframe to a mysql table
    email_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='Email',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    email_df1 = email_df.toJSON().collect()

    #returning the json
    return json.dumps(email_df1)


##To convert string to date
@app.route("/stringToDate")
def stringToDate():
    #imports
    from pyspark.sql.functions import unix_timestamp, from_unixtime

    #Converting string to date
    date_df = dataframe_mysql.select("Ship Date", from_unixtime(unix_timestamp("Ship Date", 'MM/dd/yyy')))

    #saving the dataframe to a mysql table
    date_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='Date',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    date_df1 = date_df.toJSON().collect()

    #returning the json
    return json.dumps(date_df1)


##To convert date to string
@app.route("/dateToString")
def dateToString():
    
    #imports
    from pyspark.sql.functions import date_format, col

    #Converting date to string
    string_date_df = df.select("Ship Date", date_format(col("Ship Date"), "dd-MM-YYYY"))


    #saving the dataframe to a mysql table
    string_date_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='StringDate',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    string_date_df1 = string_date_df.toJSON().collect()

    #returning the json
    return json.dumps(string_date_df1)


##To carry out text translation
@app.route("/translate")
def translate():
    lang = request.args.get("lang", None)

    #imports

    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    #function to translate text
    def trans(x):
        try:
          import goslate
          gs = goslate.Goslate()
          return gs.translate(x, lang)
        except Exception as e:
          print("The error caused is ", e)



    #Making the UDF to translate text
    trans_UDF = udf(lambda x: trans(x), StringType())

    #Translating the text
    trans_df = df.withColumn("Translated", trans_UDF("Country")).select("Country", "Translated")

    #saving the dataframe to a mysql table
    trans_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='Translate',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    trans_df1 = trans_df.toJSON().collect()

    #returning the json
    return json.dumps(trans_df1)


##To calculate Average word length
@app.route("/AvgWordLen")
def Average_Word_Length():
    #importing UDF and String Type
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    #defining a udf to convert to lower case
    def avg(x):
       try:
         words = x.split()
         average = sum(len(word) for word in words)/len(words)
         return average
       except Exception as e:
         print("The error caused is ", e)
      

    #Making the Average Word Length UDF
    AvgUdf = udf(lambda y : avg(y), StringType())

    #Finding Average Word Length
    avg_df = df.select("Country", AvgUdf("Country").alias("AvgWordLength"))

    #saving the dataframe to a mysql table
    avg_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='Average',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    avg_df1 = avg_df.toJSON().collect()

    #returning the json
    return json.dumps(avg_df1)


##To calculate word count
@app.route("/WordCount")
def Word_Count():
   
    #importing UDF and String Type
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    #defining a udf to count words
    def word_c(x):
      try:
        return len(x.split(" "))
      except Exception as e:
        print("The error caused is ", e)

    #Making the Word Count UDF
    WordCountUdf = udf(lambda y : word_c(y), StringType())

    #Finding Word Count
    WordCount_df = df.select("Country", WordCountUdf("Country").alias("Word Count"))

    #saving the dataframe to a mysql table
    WordCount_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='WordCount',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    WordCount_df1 = WordCount_df.toJSON().collect()

    #returning the json
    return json.dumps(WordCount_df1)


##To calculate Character count
@app.route("/CharCount")
def Character_Count():
    
    #importing UDF and String Type
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType


    #defining a udf to count characters
    def char_c(x):
      try:
        i = 0
        for char in x:
          if char != " ":
            i+=1
      except:
        print("The error caused is ", e)
      return i

    #Making the Word Count UDF
    CharCountUdf = udf(lambda y : char_c(y), StringType())

    #Finding Word Count
    CharCount_df = df.select("Country", CharCountUdf("Country").alias("Character Count"))

    #saving the dataframe to a mysql table
    CharCount_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='CharCount',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    CharCount_df1 = CharCount_df.toJSON().collect()

    #returning the json
    return json.dumps(CharCount_df1)


##To calculate special characters
@app.route("/Special")
def Special_Character_Count():
    #imports
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    import string

    #defining a udf to count special characters
    def sp_char_c(x):
      try:
        i = 0
        for char in x:
          if char in string.punctuation:
            i+=1
      except Exception as e:
        print("The error caused is ", e)
      return i

    #Making the Word Count UDF
    Sp_CharCountUdf = udf(lambda y : sp_char_c(y), StringType())

    #Finding Word Count
    Sp_CharCount_df = df.select("Country", Sp_CharCountUdf("Country").alias("Sp Character Count"))

    #saving the dataframe to a mysql table
    Sp_CharCount_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='SpCharCount',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    Sp_CharCount_df1 = Sp_CharCount_df.toJSON().collect()


    #returning the json
    return json.dumps(Sp_CharCount_df1)


##To remove frequent words
@app.route("/Frequent")
def Frequent_Word_Removal():
    
    #imports
    from pyspark.ml.feature import Tokenizer
    from pyspark.ml.feature import StopWordsRemover
    from pyspark.sql.functions import concat_ws

    #Frequent word removal
    tokenizer = Tokenizer(inputCol="Country", outputCol="Tokens")
    token_df = tokenizer.transform(df).select("Country", "Tokens")
    remover = StopWordsRemover()
    stopwords = remover.getStopWords()
    remover.setInputCol("Tokens")
    remover.setOutputCol("Non_freq_words")
    vector_no_freqw_df = remover.transform(token_df).select("Country", "Non_freq_words")

    #Storing result of stop words removal in the database
    vector_no_freqw_df1 = vector_no_freqw_df.withColumn('Non_freq_words', concat_ws(',', 'Non_freq_words'))

    #saving the dataframe to a mysql table
    vector_no_freqw_df1.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='RemoveFreqWords',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    vector_no_freqw_df2 = vector_no_freqw_df1.toJSON().collect()

    #returning the json
    return json.dumps(vector_no_freqw_df2)


##To extract URL from text
@app.route("/extractURL")
def urlExtract():
    
    #imports
    import re
    from pyspark.sql.types import StringType, ArrayType

    #defining a udf to extract urls
    def urll(x):
        try:
          ff = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)
          return ff
        except Exception as e:
          print("The error caused is ", e)
        

    #Making the Url UDF
    url_UDF = udf(lambda x: urll(x), StringType())

    #Extracting urls from the dataframe
    url_df = df.withColumn("URL", url_UDF("Country")).select("Country", "URL")


    #saving the dataframe to a mysql table
    url_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='ExtractUrl',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    url_df1 = url_df.toJSON().collect()

    #returning the json
    return json.dumps(url_df1)


##To remove html and xml tags
@app.route("/removeTags")
def RemoveHtmlXml():
   
    #imports

    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    import re

    #function to translate text
    def tags(x):
        try:
          cleanr = re.compile('<.*?>')
          cleantext = re.sub(cleanr, '', x)
          return cleantext
        except Exception as e:
          print("The error caused is ", e)
        

    #Making the UDF to translate text
    tags_UDF = udf(lambda x: tags(x), StringType())

    #Translating the text
    tags_df = df.withColumn("Tags Removed", tags_UDF("Country")).select("Country", "Tags Removed")

    #saving the dataframe to a mysql table
    tags_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='RemoveTags',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    tags_df1 = tags_df.toJSON().collect()

    #returning the json
    return json.dumps(tags_df1)


##To convert numbers to words
@app.route("/NumToWords")
def NumberToWords():
    
    #importing UDF and String Type
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    from num2words import num2words

    #defining a udf to convert number to word
    def num_to_word(x):
        try:
          return num2words(x)
        except Exception as e:
          print("The error caused is ",e)

    #Making the number to word UDF
    num_to_wordUdf = udf(lambda y : num_to_word(y), StringType())

    #converting the dataframe into lowercase
    num_to_word_df = df.select("Country", num_to_wordUdf("Country").alias("ToWords"))

    #saving the dataframe to a mysql table
    num_to_word_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='num2words',
    user='root',
    password='root').mode('overwrite').save()


    #converting the dataframe to json           
    num_to_word_df1 = num_to_word_df.toJSON().collect()

    #returning the json
    return json.dumps(num_to_word_df1)


##To find column Summary
@app.route("/Summary")
def ColumnSummary():
    
    #imports
    import json
    from pyspark.sql.types import IntegerType

    #Finding the count of the column
    countOfCol = df.count()

    #Finding unique elements
    unique = df.distinct().count()
    
    #Finding maximum in a column
    try:
      df1 = df.select("Country")

      if df1.dtypes[0][1] != 'int':
        df1 = df1.withColumn(df1.dtypes[0][0], df1[df1.dtypes[0][0]].cast(IntegerType()))
      
      max_element = df1.groupby().max("Country").collect()[0]#.asDict()["Country"]
    except Exception as e:
      print("The Error caused is ", e)

    #Making a triplet of the column summary as a dictionary
    summary = {
              'count'   : countOfCol,
              'maximum' : max_element[0],
              'unique'  : unique
              }

    
    #returning the json
    return json.dumps(summary)


##To carry out spell checking
@app.route("/SpellCheck")
def SpellingCheck():
    
    #imports
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StringType, ArrayType
    import enchant
    import re
    from pyspark.ml.feature import Tokenizer
    from pyspark.sql.functions import concat_ws

    #Tokenizing the column
    tokenizer = Tokenizer(inputCol="Country", outputCol="Tokens")
    token_df = tokenizer.transform(df).select("Country", "Tokens")

    #Making an enchant object
    d = enchant.Dict("en_US")


    #defining a udf for spell check
    def check_spell(in_vec):
      try:
        for word in in_vec:
          if d.check(word) == False:
            in_vec[in_vec.index(word)] = d.suggest(word)[0]
      except Exception as e:
        print("The error caused is ", e)
      return in_vec

    #Making the Spell check UDF
    spell_check_UDF = udf(lambda y : check_spell(y), ArrayType(StringType()))

    #Carry out spell check in the dataframe
    spell_check_df = token_df.select("Country", spell_check_UDF("Tokens").alias("Spell_checked"))

    #Storing result of spell check in the database
    spell_check_df1 = spell_check_df.withColumn('Spell_checked', concat_ws(' ', 'Spell_checked'))

    spell_check_df1.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='SpellChecked',
    user='root',
    password='root').mode('append').save()

    #converting the dataframe to json           
    spell_check_df2 = spell_check_df1.toJSON().collect()

    #returning the json
    return json.dumps(spell_check_df2)



##To find Data Frame Group Info
@app.route("/Group")
def GroupInfo():
    
    #imports
    import json
    from pyspark.sql.types import IntegerType
    from pyspark.sql.functions import collect_list
    from pyspark.sql.functions import concat_ws

    #Grouping the dataframe columns
    group = df.groupBy("Country").agg(collect_list("Region").alias("Region"))

    #Finding the group count
    g_count = group.count()

    #Finding maximum in the grouped column
    try:
      max_df = df.select("Region")

      if max_df.dtypes[0][1] != 'int':
        max_df = max_df.withColumn(max_df.dtypes[0][0], max_df[max_df.dtypes[0][0]].cast(IntegerType()))

      max_element = max_df.groupby().max("Region").collect()[0]#.asDict()["Country"]
    except Exception as e:
      print("The error caused is ", e)

    #Finding minimum in the grouped column
    try:
      min_df = df.select("Region")

      if min_df.dtypes[0][1] != 'int':
        min_df = min_df.withColumn(min_df.dtypes[0][0], min_df[min_df.dtypes[0][0]].cast(IntegerType()))

      min_element = min_df.groupby().min("Region").collect()[0]
    except Exception as e:
      print("The error caused is ", e)

    #Making a triplet of the column summary as a dictionary
    group_info = {
              'count'   : g_count,
              'maximum' : max_element[0],
              'minimum' : min_element[0]
              }

    #returning the json
    return json.dumps(group_info)



##For Document Term Matrix
@app.route("/DocTerm")
def DocTermFunc():
   
    #imports
    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    from pyspark.sql.functions import concat_ws, StringType

    #creating tokens of the corpus
    tokenizer = Tokenizer(inputCol="Country", outputCol="Tokens")
    tokens_df = tokenizer.transform(df).select("Country", "Tokens")

    #TF-IDF
    hashingTF = HashingTF(inputCol="Tokens", outputCol="tfFeatures", numFeatures=3)
    featurizedData = hashingTF.transform(tokens_df)

    idf = IDF(inputCol="tfFeatures", outputCol="IDFfeatures")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)


    #Storing result tf-idf in the database
    rescaledData1 = rescaledData.withColumn('Tokens', concat_ws(',', 'Tokens'))
        
    
    from pyspark.mllib.linalg import Vectors
    
    def strngfy(x):
        return Vectors.stringify(x)
    
    strng_UDF = udf(lambda y : strngfy(y), StringType())
    
    rescaledData1 = rescaledData1.withColumn("tfFeatures", strng_UDF("tfFeatures"))
    rescaledData1 = rescaledData1.withColumn("IDFfeatures", strng_UDF("IDFfeatures"))


    rescaledData1.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='DocTerm',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    rescaledData1 = rescaledData1.toJSON().collect()

    #returning the json
    return json.dumps(rescaledData1)



##To remove rare words
@app.route("/Rare")
def Rare_Word_Removal():
    
    #imports
    from pyspark.ml.feature import Tokenizer
    from pyspark.sql.functions import udf
    from pyspark.sql.types import ArrayType, StringType
    from pyspark.sql.functions import concat_ws
    from nltk.corpus import stopwords
    sw = stopwords.words("english")


    #Tokenizing
    tokenizer = Tokenizer(inputCol="Country", outputCol="Tokens")
    token_df = tokenizer.transform(df).select("Country", "Tokens")

    #Defining a function to remove rare words
    def rare(x):
      try:
        for word in x:
          if word not in sw:
            x.remove(word)
      except Exception as e:
        print("The error caused is ", e)
      return(x)


    #Making a udf for removing rare words
    rare_UDF = udf(lambda y : rare(y), ArrayType(StringType()))

    #Removing rare words
    rare_df = token_df.select("Country", rare_UDF("Tokens").alias("Non_Rare_Words"))


    #Storing result of stop words removal in the database
    rare_df1 = rare_df.withColumn('Non_Rare_Words', concat_ws(',', 'Non_Rare_Words'))

    rare_df1.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='RemoveRareWords',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    rare_df2 = rare_df1.toJSON().collect()

    #returning the json
    return json.dumps(rare_df2)



##To carry out Sentiment Analysis
@app.route("/Sentiment")
def SentimentAnalysis():
    
    #imports
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    sid = SentimentIntensityAnalyzer()

    #Defining a function to carry out sentiment analysis
    def senti(x):
      try:
        return sid.polarity_scores(x)
      except Exception as e:
        print("The error caused is ", e)

    #Making a udf for removing rare words
    sent_UDF = udf(lambda y : senti(y), StringType())

    #Removing rare words
    sent_df = df.select("Country", sent_UDF("Country").alias("Sentiment"))


    #saving the dataframe to a mysql table
    sent_df.write.format('jdbc').options(
    url='jdbc:mysql://localhost:3306/playground?useSSL=false',
    driver='com.mysql.jdbc.Driver',
    dbtable='Sentiment',
    user='root',
    password='root').mode('append').save()


    #converting the dataframe to json           
    sent_df1 = sent_df.toJSON().collect()

    #returning the json
    return json.dumps(sent_df1)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug = True)

