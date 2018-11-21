# This workbook contains
# 1. EDA - Temporal Analysis of headlines_text - Year, Month and Day
# 2. Peak values - High and lows
# 3. Feature Engineering - Word Cloud

from imports import *

base_dir = os.getcwd()
get_dir = os.path.join(base_dir,"data")

# # get_files
file_name = os.path.join(get_dir,"abcnews-date-text.csv")
df = pd.read_csv(file_name, parse_dates=[0], infer_datetime_format=True)
df.head(10)
df.info()

# feature engineering:
df = pd.read_csv(file_name,dtype={'publish_date': object})
df['publish_month'] = df.publish_date.str[:6]
df['publish_year'] = df.publish_date.str[:4]
df['publish_month_only'] = df.publish_date.str[4:6]
df['publish_day_only'] = df.publish_date.str[6:8]
df['dt_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')
df['dt_month'] = pd.to_datetime(df['publish_month'], format='%Y%m')


# Monthly and daily plot
grp_month = df.groupby(['dt_month'])['headline_text'].count()
ts = pd.Series(grp_month,index = grp_month.index)
ts.plot(kind='line', figsize=(20,10),title='Number of headlines per month',)

a = pd.DatetimeIndex(start='2003-02-01',end='2017-12-31',freq='BM')
fig, ax = plt.subplots()
ax.plot_date(a.to_pydatetime(), ts, 'v-')
ax.xaxis.set_minor_formatter(dates.DateFormatter('%d%a'))
ax.xaxis.grid(True, which="minor")
ax.yaxis.grid()
ax.xaxis.set_major_locator(dates.MonthLocator())
ax.xaxis.set_major_formatter(dates.DateFormatter('%b%Y'))
plt.xticks(rotation='vertical')
plt.tick_params(labelsize=5)
plt.show()


# We can see that there is super high peak on Aug 2012
# Lets see what happened on Aug 2012

year2012 = df[df['publish_year'] == '2012']
grp_months = year2012.groupby(['publish_month'])['headline_text'].count()
print("Average Number of news headlines for the year 2012 : %d " %grp_months.mean())
ts2 = pd.Series(grp_months.tolist(),index=pd.date_range('2012-01-01', periods=12))
ts2.plot(kind='bar',figsize=(10,7),title='Year 2012 - Number of headlines per day')


# On an average, the highest number of articles published on this month - Aug 2012 --- 288
# From 20th Aug to 24th Aug 2012,  there is a strong event occurred

aug2012 = df[df['dt_month'] == '2012-08']
grp_day = aug2012.groupby(['publish_day_only'])['headline_text'].count()
print("Max. Number of news headlines for the month Aug 2012 : %d " %grp_day.max())
print()
ts2 = pd.Series(grp_day.tolist(),index=pd.date_range('2012-08-01', periods=31))
ts2.plot(kind='bar',figsize=(10,7),title='August 2012 - Number of headlines per day')

# word Cloud
titles = aug2012["headline_text"].values
stopWords = set(stopwords.words('english'))
titles_token = [word for word in titles if word not in stopWords ]
wordcloud = WordCloud(max_font_size=40).generate(str(titles_token))
plt.close('all')
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# From the exploratory analysis and the word cloud visualisation, we can understand that
# the General Elections and Syrian War are the major prime factors for the upward spike in the trend.





