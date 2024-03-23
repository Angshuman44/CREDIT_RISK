from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app=application

#Route for Home

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            car_owner=request.form.get('car_owner'),
            property_owner=request.form.get('property_owner'),
            children=int(request.form.get('children')),
            annual_income=float(request.form.get('annual_income')),
            type_income=request.form.get('type_income'),
            education=request.form.get('education'),
            marital_status=request.form.get('marital_status'),
            housing_type=request.form.get('housing_type'),
            age_years=float(request.form.get('age_years')),
            years_employed=float(request.form.get('years_employed')),
            family_members=int(request.form.get('family_members'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        print(results)
        if results[0] == 0:
            verdict = 'Approved'
        else:
            verdict = 'Rejected'

        return render_template('home.html', predicted_result=verdict)





if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)   

    













# def predict_datapoint():
#     if request.method=='GET':
#         return render_template('home.html')
#     else:
#         data=CustomData(
#             gender=request.form.get('gender'),
#             race_ethnicity=request.form.get('ethnicity'),
#             parental_level_of_education=request.form.get('parental_level_of_education'),
#             lunch=request.form.get('lunch'),
#             test_preparation_course=request.form.get('test_preparation_course'),
#             reading_score=float(request.form.get('writing_score')),
#             writing_score=float(request.form.get('reading_score'))

#         )
#         pred_df=data.get_data_as_data_frame()
#         print(pred_df)
#         print("Before Prediction")

#         predict_pipeline=PredictPipeline()
#         print("Mid Prediction")
#         results=predict_pipeline.predict(pred_df)
#         print("after Prediction")
#         print(results)

#         return render_template('home.html',predicted_result=results[0])
    
     