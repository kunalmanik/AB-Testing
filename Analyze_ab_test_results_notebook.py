
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv("ab_data.csv")
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[3]:


print("Number of rows : " + str(df.shape[0]))


# c. The number of unique users in the dataset.

# In[4]:


print("Number of unique users : " + str(df.user_id.nunique()))


# d. The proportion of users converted.

# In[5]:


print("Converted proportion : " + str(df.converted.mean()))


# e. The number of times the `new_page` and `treatment` don't line up.

# In[6]:


qr1 = df.query("group == 'control' and landing_page == 'new_page'")
qr2 = df.query("group == 'treatment' and landing_page == 'old_page'")
print("The number of times the new_page and treatment don't line up : " + str(len(qr1) + len(qr2)))


# f. Do any of the rows have missing values?

# In[7]:


df.info()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


df.drop(qr1.index, inplace=True)
df.drop(qr2.index, inplace=True)


# In[9]:


df.info()


# In[10]:


df.to_csv('ab_edited.csv', index=False)


# In[11]:


df2 = pd.read_csv('ab_edited.csv')


# In[12]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[13]:


print("Number of Unique ids in df2 : " + str(df2.user_id.nunique()))


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[14]:


df2[df2.duplicated(['user_id'], keep=False)]['user_id']


# c. What is the row information for the repeat **user_id**? 

# In[15]:


df2[df2.duplicated(['user_id'], keep=False)]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[16]:


df2 = df2.drop(1876)


# In[17]:


df2.info()


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[18]:


df2.converted.mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[19]:


df2.groupby('group').describe()


# Given that an individual was in the control group, the probability they converted is 0.120386

# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# Given that an individual was in the treatment group, the probability they converted is 0.118808

# d. What is the probability that an individual received the new page?

# In[20]:


new_page_p = len(df.query("group == 'treatment'"))/df.shape[0]
print("The probability that an individual received the new page : " + str(new_page_p))


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# 1. Given that an individual was in the **treatment** group, the probability they converted is **0.118808**
# 2. Given that an individual was in the **control** group, the probability they converted is **0.120386**    
# 3. Based on the information above, it can be concluded that there is not much to differentiate the performance of both the pages. Subsequently,we can say that there not much evidence yet to state the above.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# H<sub>0</sub> : P<sub>old</sub> >= P<sub>new</sub> <br><br>
# H<sub>1</sub> : P<sub>old</sub> < P<sub>new</sub>

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[21]:


p_new = df2.converted.mean()
print(p_new)


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[22]:


p_old = df2.converted.mean()
print(p_old)


# c. What is $n_{new}$?

# In[23]:


n_new = len(df2.query("group == 'treatment'"))
print(n_new)


# d. What is $n_{old}$?

# In[24]:


n_old = len(df2.query("group == 'control'"))
print(n_old)


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[25]:


new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_new, (1-p_new)])


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[26]:


old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_old, (1-p_old)])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[27]:


p_diff = new_page_converted[:145274]/n_new - old_page_converted/n_old
print(p_diff)


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[28]:


p_diffs = []

for _ in range(10000):
    new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_new, (1-p_new)]).mean()
    old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_old, (1-p_old)]).mean()
    diff = new_page_converted - old_page_converted 
    p_diffs.append(diff)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[29]:


plt.hist(p_diffs)
plt.xlabel('p_diffs')
plt.ylabel('Frequency')
plt.title('Plot of simulated p_diffs');


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[30]:


actual_diff = df2[df2.group == 'treatment']['converted'].mean() -  df2[df2.group == 'control']['converted'].mean()
actual_diff


# In[31]:


p_diffs = np.array(p_diffs)
p_diffs


# In[32]:


(p_diffs > actual_diff).mean()


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# We are computing p-values here, that is the probability of observing our statistic (or one more extreme in favor of the alternative) if the null hypothesis is true.
#     Here, we find that there is no advantage associated with conversion of new pages. We conclude that null hypothesis is true as old and new pages perform almost similarly.
# 

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[33]:


import statsmodels.api as sm

convert_old = sum(df2.query("group == 'control'")['converted'])
convert_new = sum(df2.query("group == 'treatment'")['converted'])
n_old = len(df2.query("group == 'control'"))
n_new = len(df2.query("group == 'treatment'"))

print(convert_old, convert_new, n_old, n_new)


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[34]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
print(z_score, p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# In[35]:


from scipy.stats import norm

print(norm.cdf(z_score))
# Tells us how significant our z-score is for our single-sides test, assumed at 95% confidence level
print(norm.ppf(1-(0.05)))
# Tells us what our critical value at 95% confidence is 
# Here, we take the 95% values as specified in PartII.


# <h2>Answer</h2>
# <ol>
#     <li>We find that the z-score of 1.31092419842 is less than the critical value of 1.64485362695. So, we accept the null hypothesis.</li>
#     <li>We find that old pages are only minutely better than new pages.</li>    
# </ol>

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Logistic Regression**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[36]:


df['intercept']=1
df[['control', 'treatment']] = pd.get_dummies(df['group'])


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[37]:


logit_stats = sm.Logit(df['converted'],df[['intercept','treatment']])


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[38]:


results = logit_stats.fit()
results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# (a) <br><br>
# 
# H<sub>0</sub> : P<sub>old</sub> - P<sub>new</sub> = 0 <br><br>
# H<sub>1</sub> : P<sub>old</sub> - P<sub>new</sub> != 0

# (b) The test in <b>Part II</b> was <i>one sided</i> while the current one is <i>Two sided Test</i>.
# 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Advantages**
# We should consider other factors into the regression model as they might influence the conversions too.
# 
# **Disadvantages**
# The disadvantage of adding additional terms into the regression model is that even with additional factors we can never account for all influencing factors.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[39]:


countries_df = pd.read_csv('./countries.csv')
countries_df.head()


# In[40]:


df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new.head()


# In[41]:


df_new['country'].value_counts()


# In[42]:


df_new[['CA', 'US']] = pd.get_dummies(df_new['country'])[['CA','US']]

df_new['country'].astype(str).value_counts()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[43]:


#Fitting the Linear Model
df['intercept'] = 1

log_mod = sm.Logit(df_new['converted'], df_new[['CA', 'US']])
results = log_mod.fit()
results.summary()


# In[44]:


np.exp(results.params)


# In[45]:


df.groupby('group').mean()['converted']


# **We find, using this model, that the values do not show a significant difference in the conversion rates for control group and treatment group. Thus we can accept the Null Hypothesis.**

# <h1>Project Conclusion</h1>
# <ul>
# <li>The performance of the both the pages were mostly similar with old_page being better by a very small margin.</li>
#     <li>We accept the Null Hypothesis(H<sub>0</sub>) and reject the Alternate Hypothesis(H<sub>1</sub>)</li>
# </ul>

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! 
# 
# ### Gather Submission Materials
# 
# Once you are satisfied with the status of your Notebook, you should save it in a format that will make it easy for others to read. You can use the __File -> Download as -> HTML (.html)__ menu to save your notebook as an .html file. If you are working locally and get an error about "No module name", then open a terminal and try installing the missing module using `pip install <module_name>` (don't include the "<" or ">" or any words following a period in the module name).
# 
# You will submit both your original Notebook and an HTML or PDF copy of the Notebook for review. There is no need for you to include any data files with your submission. If you made reference to other websites, books, and other resources to help you in solving tasks in the project, make sure that you document them. It is recommended that you either add a "Resources" section in a Markdown cell at the end of the Notebook report, or you can include a `readme.txt` file documenting your sources.
# 
# ### Submit the Project
# 
# When you're ready, click on the "Submit Project" button to go to the project submission page. You can submit your files as a .zip archive or you can link to a GitHub repository containing your project files. If you go with GitHub, note that your submission will be a snapshot of the linked repository at time of submission. It is recommended that you keep each project in a separate repository to avoid any potential confusion: if a reviewer gets multiple folders representing multiple projects, there might be confusion regarding what project is to be evaluated.
# 
# It can take us up to a week to grade the project, but in most cases it is much faster. You will get an email once your submission has been reviewed. If you are having any problems submitting your project or wish to check on the status of your submission, please email us at dataanalyst-project@udacity.com. In the meantime, you should feel free to continue on with your learning journey by continuing on to the next module in the program.

# **<h1>Resources and References</h1>**

# <ul>
#     <li>Udacity Nanodegree Videos and Resources</li>
# <li>https://stackoverflow.com/questions/26244309/how-to-analyze-all-duplicate-entries-in-this-pandas-dataframe</li>
# <li>https://stats.stackexchange.com/questions/52067/does-adding-more-variables-into-a-multivariable-regression-change-coefficients-o</li>
# </ul>
