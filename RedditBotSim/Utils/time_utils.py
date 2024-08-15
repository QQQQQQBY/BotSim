import datetime
import time
import calendar
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def time_string_to_timestamp(time_string):
    """
    Replace a time string with a timestamp
    :param time_string: Time character string. The format is TIME_FORMAT
    :return: float, the timestamp in seconds
    """
    return time.mktime(time.strptime(time_string, TIME_FORMAT))


def time_format_change(time_string, format1, format2):
    """
    Replace a time string with a timestamp
    :param time_string: Time character string. The format is TIME_FORMAT
    :return: float, the timestamp in seconds
    """
    timestamp = time.mktime(time.strptime(time_string, format1))
    new_time_string = time.strftime(format2, time.localtime(timestamp))
    return new_time_string


def timestamp_to_time_string(timestamp):
    """
    The timestamp is converted to a time string
    :param timestamp: float, the timestamp in seconds
    :return: string Indicates the time string. The format is TIME_FORMAT
    """
    time_local = time.localtime(timestamp)
    return time.strftime(TIME_FORMAT, time_local)


def get_now_time():
    """
    Gets the time string for the current time
    :return: string Indicates the time string. The format is TIME_FORMAT
    """
    return datetime.datetime.now().strftime(TIME_FORMAT)


def get_now_day():
    """
    Gets the time string for the current time
    :return: string Indicates the time string. The format is TIME_FORMAT
    """
    now_time = datetime.datetime.now().strftime(TIME_FORMAT)
    return now_time.split(' ')[0]


def gap_times(start, days, hours, minutes):
    """
    Gets the time from the start time to the specified number of days
    :param start: string indicates the specified start time
    :param days: int indicates the number of days calculated backwards
    :return: string Indicates the start time and end time
    """
    try:
        start_time = datetime.datetime.strptime(start, TIME_FORMAT)
        
        end_time = (start_time + datetime.timedelta(days=days, hours=hours, minutes=minutes)).strftime(TIME_FORMAT)
        end_time_split = end_time.split(' ')[0]
        year, month, day = end_time_split.split('-')
        year, month, day = int(year), int(month), int(day)
        last_day_of_month = calendar.monthrange(year, month)[1]
        if day <= calendar.monthrange(year, month)[1]:
            return start_time.strftime(TIME_FORMAT), end_time
        if day > last_day_of_month:
            day = last_day_of_month    
            end_time = datetime.datetime(year, month, day)
            end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')       
            return start_time.strftime(TIME_FORMAT), end_time
    except Exception as e:
        print("======Exception====== /n", e)
        return 

def gap_minutes(start, minutes):
    """
    Gets the time from the start time to the specified number of days
    :param start: string indicates the specified start time
    :return: string Indicates the start time and end time
    """
    start_time = datetime.datetime.strptime(start, TIME_FORMAT)
    end_time = (start_time + datetime.timedelta(gap_day)).strftime(TIME_FORMAT)
    return start_time, end_time