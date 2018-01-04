from flask import Flask,Response,render_template
from flask import request
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import pickle
path = os.getcwd()
print(path)
port = int(os.getenv("PORT", 3000))
upload_folder = path
ALLOWED_EXTENSIONS = set(['pkl','txt','csv'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder

@app.route('/upload')
def redirect():
    return render_template('upload1.html')

@app.route('/uploader', methods = ['GET','POST'])
def upload_file():
    try:
        f = request.files['file']
        f.save(secure_filename(f.filename))
        print('file uploaded successfully')
        return 'file uploaded successfully'
    except Exception as e:
        print(e)

@app.route('/train',methods=['GET','POST'])
def train_data():
    try:
        data = pd.read_csv('./data.csv')
        print("data.shape", data.shape)
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
        df_copy = train_set.copy()
        print("df_copy", df_copy)
        test_set_full = test_set.copy()
        print("test_set_full", test_set_full)
        lin_reg = LinearRegression()
        print("linear model imported")

        column_data = pd.read_csv('./columns.csv')
        column_1 = (column_data.columns[0])
        print("column 1", column_1)  # size
        column_2 = (column_data.columns[1])
        print("column 2", column_2)  # bedrooms
        column_3 = (column_data.columns[2])
        print("column 3", column_3)  # age
        column_4 = (column_data.columns[3])
        print("column 4", column_4)  # bathrooms
        column_5 = (column_data.columns[4])
        print("column 5", column_5)  # Price

        test_set = test_set.drop["Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]
        train_set = train_set.drop["Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"]
        print("train_set",train_set)
        print("test_set",test_set)


        # For 1st variable
        test_set = test_set.drop([column_1], axis=1)
        print("test_set", test_set)
        train_labels = train_set[column_1]
        print("train_lables", train_labels)
        train_set_full = train_set.copy()
        train_set = train_set.drop([column_1], axis=1)
        lin_reg.fit(train_set, train_labels)
        print("linear fit done 1")
        pickle.dump(lin_reg, open('predict1.pkl', 'wb'))

        # For 2nd variable
        test_set = test_set.drop([column_2], axis=1)
        print("test_set", test_set)
        train_labels = train_set[column_2]
        print("train_lables", train_labels)
        train_set_full = train_set.copy()
        train_set = train_set.drop([column_2], axis=1)
        lin_reg.fit(train_set, train_labels)
        print("linear fit done 2")
        pickle.dump(lin_reg, open('predict2.pkl', 'wb'))

        # For 3rd variable
        test_set = test_set.drop([column_3], axis=1)
        print("test_set", test_set)
        train_labels = train_set[column_3]
        print("train_lables", train_labels)
        train_set_full = train_set.copy()
        train_set = train_set.drop([column_3], axis=1)
        lin_reg.fit(train_set, train_labels)
        print("linear fit done 3")
        pickle.dump(lin_reg, open('predict3.pkl', 'wb'))

        # For 4th variable
        test_set = test_set.drop([column_4], axis=1)
        print("test_set", test_set)
        train_labels = train_set[column_4]
        print("train_lables", train_labels)
        train_set_full = train_set.copy()
        train_set = train_set.drop([column_4], axis=1)
        lin_reg.fit(train_set, train_labels)
        print("linear fit done")
        pickle.dump(lin_reg, open('predict4.pkl', 'wb'))

        # For 5th variable
        test_set = test_set.drop([column_5], axis=1)
        print("test_set", test_set)
        train_labels = train_set[column_5]
        print("train_lables", train_labels)
        train_set_full = train_set.copy()
        train_set = train_set.drop([column_5], axis=1)
        lin_reg.fit(train_set, train_labels)
        print("linear fit done")
        pickle.dump(lin_reg, open('predict5.pkl', 'wb'))
    except Exception as e:
        print(e)
