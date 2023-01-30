import twint
import nest_asyncio 
nest_asyncio.apply()


class Scrapper():
    def __init__(self, Lang, Search, Since, Until, Limit, Images=True, Pandas=True):
        self.Lang=Lang
        self.Search=Search
        self.Since=Since
        self.Until=Until
        self.Limit=Limit
        self.Images=Images     
        self.Pandas=Pandas
    
    def scrap(self):
        c = twint.Config() 
        c.Lang=self.Lang      #specify the language of the tweet you want to scrape "en" for English
        c.Search=self.Search  #fill in the query that you want to search Eg "Coronavirus"
        c.Since=self.Since    #give a specific time for the date of the tweet that will be scraped
        c.Until=self.Until    #give a specific time for the end-date of the tweet that will be scraped
        
        c.Limit=self.Limit    #Limit the number of tweets that are scraped
        c.Images=self.Images  #Include images as well
        c.Pandas=self.Pandas  #save the output data in pandas dataframe
        
        twint.run.Search(c)
        Tweets_df=twint.storage.panda.Tweets_df
        return Tweets_df
    def run(self):
        data_fr=self.scrap()
        #data_fr=data_fr[['id', 'tweet', 'Language', 'hashtags', 'thumbnail']]
        #data_fr=data_fr[data_fr['Language']=='en']
        return data_fr.to_csv('corona_data.csv')
