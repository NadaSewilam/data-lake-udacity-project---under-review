import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, monotonically_increasing_id
from pyspark.sql import Window

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    ## """Creates a spark session and configure it """
    spark = SparkSession.builder.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0").getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Loads song data into spark df 
    then extract songs and artists tables then move them to the S3 bucket
    Args:
        spark (spark session): session to use spark
        input_data (str): path to input data
        output_data (str): path to output data
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.selectExpr('song_id', \
                                'title', \
                                'artist_id', \
                                'year', \
                                'duration').orderBy("song_id").drop_duplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.songs_table.write.partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs.parquet'), 'overwrite')
    
    # extract columns to create artists table
    artists_table = df.selectExpr('artist_id', \
                              'artist_name', \
                              'artist_location', \
                              'artist_latitude', \
                              'artist_longitude').dropDuplicates(['artist_id'])
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artists.parquet'), 'overwrite')


def process_log_data(spark, input_data, output_data):
    """Loads log data into spark df
    then extract users, time and songplays tables then move them to s3 bucket
    Args:
        spark (spark session): session to use spark
        input_data (str): path to input data
        output_data (str): path to output data
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*/*.json')

    # read log data file
    df =  spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    artists_table = df['userId', 'firstName', 'lastName',
                     'gender', 'level'].dropDuplicates(['userId'])
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users.parquet'), 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime =  udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000.0)))
    df = df.withColumn("datetime", get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.select(
        col('datetime').alias('start_time'),
        hour('datetime').alias('hour'),
        dayofmonth('datetime').alias('day'),
        weekofyear('datetime').alias('week'),
        month('datetime').alias('month'),
        year('datetime').alias('year')
    ).dropDuplicates(['start_time'])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').parquet(
        os.path.join(output_data, 'time.parquet'), 'overwrite')

    # read in song data to use for songplays table -SAME AS LINE 54
    song_df = os.path.join(input_data, 'song_data/*/*/*/*.json')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table =  df.select(
        col('ts').alias('ts'),
        col('sessionId').alias('session_id'),
        col('userId').alias('user_id'),
        col('artist_id').alias('artist_id'),
        col('song_id').alias('song_id'),
        col('level').alias('level'),          
        col('location').alias('location'),
        col('userAgent').alias('user_agent'),
        month('datetime').alias('month'),
        col('year').alias('year')      
    ).withColumn('songplay_id', monotonically_increasing_id())


    # write songplays table to parquet files partitioned by year and month - SAME AS LINE 88
    songplays_table.write.partitionBy('year', 'month').parquet(
        os.path.join(output_data, 'songplays.parquet'), 'overwrite')


def main():
    """Initiate spark session then process song data then process log data"""
    
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3://nadabucket19/proj4/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
