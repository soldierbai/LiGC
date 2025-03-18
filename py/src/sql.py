import pymysql

def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='<PASSWORD>',
        db='test',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
    )