import mysql.connector
from mysql.connector import errorcode
import base64


# This class handles all connections and sql-queries to the AWS database
class Database:
    # define class variables
    db = mysql.connector
    #cursor = db.cursor()
    
    #   option file for credentials ???
    user=""
    host=""
    port=""
    database=""
    
    ### Connection to database. Put credentials in option file?? ###
    def connect():
    
    #while(!db.connect)  while not connected, keep trying...
        print("connecting to database...")
        try :
            db.connect(user="lojpan",
                        password="L0jpan1.",
                        host="mydatabase.c3mtrqj0btdn.us-east-1.rds.amazonaws.com",
                        port="3306",
                        database="Object Detection")

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                    print("ERROR: Wrong user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("ERROR: Database does not exist")
            else:
                print(err)
        else:
                print("Successfully connected to DB!")
                #break  when connected exit while-loops
    
    # Create a cursor. Requires a mysql.connector object as input parameter
    def createCursor(database):
        return database.cursor()
                
    def read_img():
         ### Open image to encode
        with open('/home/pi/Desktop/snapshots/' + object_name + str(1) + '.jpg', 'rb') as f:
            photo = f.read()
        return photo

    # Insert data to database
    def insertData(label, img, cursor):
        # Read img from folder
        img = read_img()

        # Create sql-query, first a table is selected then names of colunmns to insert data into and then insert actual values. Timestamp uses NOW() to generate current date and time for every post. Images need to be encoded before uploded to be able to read it properly later. Base64 is used for encode/decode
        sql = "insert into Objects (id, description, count, timestamp, image) VALUES('id', %s , 1, NOW(), %s)"
        val = (object_name, base64.b64encode(img))
        # Execute query
        cursor.execute(sql, val)
        # Commit to changes in database
        db.commit()

    # Images need to be encoded to a binary string (of bytes) before uplod to be able to decode and read it properly later. Base64 is used for encode/decode
    def encode_img(img):
        encoded_img = base64.b64encode(photo)
        return encode_img
    
    # Decode (binary)
    def decode_img():
        ### Decode fetched data
        data1= base64.b64decode(data[0][0])

        ### Create file to write to
        with open('/home/pi/Desktop/snapshots/Download_img.jpg', 'wb') as file_to_save:
            file_to_save.write(data1)
    
    def download_img():
        #### Select image to decode
        sql= "select image from Objects where id = 133"
        ### Execute query
        cursor.execute(sql)
        ### Fetch selected data
        data = cursor.fetchall()
        ### Close connection
        db.close()

    def close(database):
        database.close()
        




