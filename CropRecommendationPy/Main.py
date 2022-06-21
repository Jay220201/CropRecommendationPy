from flask import Flask, render_template, flash, request, session,send_file
from flask import render_template, redirect, url_for, request
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from werkzeug.utils import secure_filename
import datetime
import mysql.connector
import sys

import pickle


import numpy as np


app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/")
def homepage():

    return render_template('index.html')

@app.route("/AdminLogin")
def AdminLogin():

    return render_template('AdminLogin.html')


@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')

@app.route("/NewUser")
def NewUser():
    return render_template('NewUser.html')

@app.route("/NewQuery1")
def NewQuery1():
    return render_template('NewQueryReg.html')

@app.route("/UploadDataset")
def UploadDataset():
    return render_template('ViewExcel.html')



@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb ")
    data = cur.fetchall()
    return render_template('AdminHome.html',data=data)






@app.route("/UserHome")
def UserHome():
    user = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb where username='" + user + "'")
    data = cur.fetchall()
    return render_template('UserHome.html',data=data)


@app.route("/UQueryandAns")
def UQueryandAns():

    uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult='waiting'")
    data = cur.fetchall()

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult !='waiting'")
    data1 = cur.fetchall()


    return render_template('UserQueryAnswerinfo.html', wait=data, answ=data1 )


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
       if request.form['uname'] == 'admin' or request.form['password'] == 'admin':

           conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
           # cursor = conn.cursor()
           cur = conn.cursor()
           cur.execute("SELECT * FROM regtb ")
           data = cur.fetchall()
           return render_template('AdminHome.html' , data=data)

       else:
        return render_template('index.html', error=error)


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():

    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            alert = 'Username or Password is wrong'
            render_template('goback.html', data=alert)



        else:
            print(data[0])
            session['uid'] = data[0]
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username + "' and Password='" + password + "'")
            data = cur.fetchall()

            return render_template('UserHome.html', data=data )




@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':

        name1 = request.form['name']
        gender1 = request.form['gender']
        Age = request.form['age']
        email = request.form['email']
        pnumber = request.form['phone']
        address = request.form['address']

        uname = request.form['uname']
        password = request.form['psw']


        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO regtb VALUES ('" + name1 + "','" + gender1 + "','" + Age + "','" + email + "','" + pnumber + "','" + address + "','" + uname + "','" + password + "')")
        conn.commit()
        conn.close()
        # return 'file register successfully'


    return render_template('UserLogin.html')



@app.route("/newquery", methods=['GET', 'POST'])
def newquery():
    if request.method == 'POST':
        uname = session['uname']
        nitrogen = request.form['nitrogen']
        phosphorus = request.form['phosphorus']
        potassium = request.form['potassium']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']




        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Querytb VALUES ('','" + uname + "','" + nitrogen + "','" + phosphorus + "','" + potassium + "','"+temperature+"','"+humidity +"','"+ ph
            +"','"+ rainfall +"','waiting','')")
        conn.commit()
        conn.close()
        # return 'file register successfully'
        uname = session['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        # cursor = conn.cursor()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult='waiting'")
        data = cur.fetchall()

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
        # cursor = conn.cursor()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Querytb where UserName='" + uname + "' and DResult !='waiting'")
        data1 = cur.fetchall()

        return render_template('UserQueryAnswerinfo.html', wait=data, answ=data1)


@app.route("/excelpost", methods=['GET', 'POST'])
def uploadassign():
    if request.method == 'POST':


        file = request.files['fileupload']
        file_extension = file.filename.split('.')[1]
        print(file_extension)
        #file.save("static/upload/" + secure_filename(file.filename))

        import pandas as pd
        import matplotlib.pyplot as plt
        df = ''
        if file_extension == 'xlsx':
            df = pd.read_excel(file.read(), engine='openpyxl')
        elif file_extension == 'xls':
            df = pd.read_excel(file.read())
        elif file_extension == 'csv':
            df = pd.read_csv(file)



        print(df)




        import seaborn as sns
        sns.countplot(df['label'], label="Count")
        plt.savefig('static/images/out.jpg')
        iimg = 'static/images/out.jpg'

        #plt.show()


        print(df)


        # import pandas as pd
        import matplotlib.pyplot as plt

        # read-in data
        # data = pd.read_csv('./test.csv', sep='\t')

        import seaborn as sns
        sns.countplot(df['label'], label="Count")
        plt.show()

        df.label = df.label.map({'rice': 0,
                                 'maize': 1,
                                 'chickpea': 2,
                                 'kidneybeans': 3,
                                 'pigeonpeas': 4,
                                 'mothbeans': 5,
                                 'mungbean': 6,
                                 'blackgram': 7,
                                 'lentil': 8,
                                 'pomegranate': 9,
                                 'banana': 10,
                                 'mango': 11,
                                 'grapes': 12,
                                 'watermelon': 13,
                                 'muskmelon': 14,
                                 'apple': 15,
                                 'orange': 16,
                                 'papaya': 17,
                                 'coconut': 18,
                                 'cotton': 19,
                                 'jute': 20,
                                 'coffee': 21})

        # Replacing the 0 values by NaN
        df_copy = df.copy(deep=True)
        df_copy[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] = df_copy[
            ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].replace(0, np.NaN)

        # Model Building
        from sklearn.model_selection import train_test_split
        df.drop(df.columns[np.isnan(df).any()], axis=1)
        X = df.drop(columns='label')
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)




        from sklearn.metrics import classification_report
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=0)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        clreport = classification_report(y_test, y_pred)

        print("Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train)))
        print("Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test)))

        Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
        Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))



        # Creating a pickle file for the classifier
        filename = 'crop-prediction-rfc-model.pkl'
        pickle.dump(classifier, open(filename, 'wb'))



        print("Training process is complete Model File Saved!")

        df= df.head(200)




        return render_template('ViewExcel.html', data=df.to_html(), dataimg=iimg ,tacc=Tacc,testacc=Testacc,report=clreport)


@app.route("/AdminQinfo")
def AdminQinfo():

    #uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult='waiting'")
    data = cur.fetchall()


    return render_template('AdminQueryInfo.html', data=data )


@app.route("/answer")
def answer():

    Answer = ''
    Prescription=''
    id =  request.args.get('lid')

    msssg="Prediction"


    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    cursor = conn.cursor()
    cursor.execute("SELECT  *  FROM Querytb where  id='" + id + "'")
    data = cursor.fetchone()

    if data:
        UserName = data[1]
        nitrogen = data[2]
        phosphorus = data[3]
        potassium = data[4]
        temperature = data[5]
        humidity = data[6]
        ph = data[7]
        rainfall = data[8]



    else:
        return 'Incorrect username / password !'

    nit = float(nitrogen)
    pho = float(phosphorus)
    po = float(potassium)
    te = float(temperature)
    hu = float(humidity)
    phh = float(ph)
    ra = float(rainfall)
    #age = int(age)

    filename = 'crop-prediction-rfc-model.pkl'
    classifier = pickle.load(open(filename, 'rb'))

    data = np.array([[nit, pho, po, te, hu, phh, ra ]])
    my_prediction = classifier.predict(data)
    print(my_prediction)
    crop=''


    if my_prediction == 0:
        Answer = 'Predict'
        crop = 'rice'

    elif my_prediction == 1:
        Answer = 'Predict'
        crop = 'rice'
    elif my_prediction == 2:
        Answer = 'Predict'
        crop = 'chickpea'
    elif my_prediction == 3:
        Answer = 'Predict'
        crop = 'kidneybeans'
    elif my_prediction == 4:
        Answer = 'Predict'
        crop = 'pigeonpeas'
    elif my_prediction == 5:
        Answer = 'Predict'
        crop = 'mothbeans'
    elif my_prediction == 6:
        Answer = 'Predict'
        crop = 'mungbean'
    elif my_prediction == 7:
        Answer = 'Predict'
        crop = 'blackgram'
    elif my_prediction == 8:
        Answer = 'Predict'
        crop = 'lentil'
    elif my_prediction == 9:
        Answer = 'Predict'
        crop = 'pomegranate'
    elif my_prediction == 10:
        Answer = 'Predict'
        crop = 'banana'
    elif my_prediction == 11:
        Answer = 'Predict'
        crop = 'mango'
    elif my_prediction == 12:
        Answer = 'Predict'
        crop = 'grapes'
    elif my_prediction == 13:
        Answer = 'Predict'
        crop = 'watermelon'
    elif my_prediction == 14:
        Answer = 'Predict'
        crop = 'muskmelon'
    elif my_prediction == 15:
        Answer = 'Predict'
        crop = 'apple'
    elif my_prediction == 16:
        Answer = 'Predict'
        crop = 'orange'
    elif my_prediction == 17:
        Answer = 'Predict'
        crop = 'papaya'
    elif my_prediction == 18:
        Answer = 'Predict'
        crop = 'coconut'
    elif my_prediction == 19:
        Answer = 'Predict'
        crop = 'cotton'
    elif my_prediction == 20:
        Answer = 'Predict'
        crop = 'jute'
    elif my_prediction == 21:
        Answer = 'Predict'
        crop = 'coffee'



    else:
        Answer = 'Predict'

        crop='Crop info not Found!'



    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    cursor = conn.cursor()
    cursor.execute(
        "update Querytb set DResult='"+Answer+"', CropInfo='" + crop +"'  where id='" + str(id) + "' ")
    conn.commit()
    conn.close()

    msssg =msssg + " :"+ crop

    conn3 = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    cur3 = conn3.cursor()
    cur3.execute("SELECT * FROM regtb where 	UserName='" + str(UserName) + "'")
    data3 = cur3.fetchone()
    if data3:
        phnumber = data3[4]
        print(phnumber)
        sendmsg(phnumber, msssg)

    # return 'file register successfully'
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult !='waiting '")
    data = cur.fetchall()
    return render_template('AdminAnswer.html', data=data)



def sendmsg(targetno,message):
    import requests
    requests.post("http://smsserver9.creativepoint.in/api.php?username=fantasy&password=596692&to=" + targetno + "&from=FSSMSS&message=Dear user  your msg is " + message + " Sent By FSMSG FSSMSS&PEID=1501563800000030506&templateid=1507162882948811640")


@app.route("/AdminAinfo")
def AdminAinfo():



    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1croprecomdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Querytb where  DResult !='waiting'")
    data = cur.fetchall()


    return render_template('AdminAnswer.html', data=data )


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)