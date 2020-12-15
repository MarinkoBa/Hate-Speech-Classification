import json
import os

from src.utils.get_tweets_by_id import get_tweets_by_id



if __name__ == "__main__":  
    # load configuration fields
    config = json.load(open('./config.json', 'r'))
    

    df = get_tweets_by_id(config,
                          os.path.join('data', 'NAACL_SRW_2016.csv'))