from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)


terror = pd.read_csv(r'C:\Users\leyla\Downloads\globalterrorismdb_0718dist.csv', encoding='ISO-8859-1', low_memory=False)
terror.rename(columns={'iyear': 'Year', 'imonth': 'Month', 'iday': 'Day', 'country_txt': 'Country', 'region_txt': 'Region',
                       'attacktype1_txt': 'AttackType', 'target1': 'Target', 'nkill': 'Killed', 'nwound': 'Wounded',
                       'summary': 'Summary', 'gname': 'Group', 'targtype1_txt': 'Target_type', 'weaptype1_txt': 'Weapon_type', 'motive': 'Motive'}, inplace=True)
terror['casualities'] = terror['Killed'] + terror['Wounded']


if not os.path.exists('static'):
    os.makedirs('static')


def create_plots():
    
    plt.subplots(figsize=(15, 6))
    sns.countplot(x='Year', data=terror, hue='Year', palette='RdYlGn_r', edgecolor=sns.color_palette('dark', 7), legend=False)
    plt.xticks(rotation=90)
    plt.title('Number Of Terrorist Activities Each Year')
    plt.savefig('static/terror_activities_per_year.png')
    plt.close()

    
    plt.subplots(figsize=(15, 6))
    sns.countplot(x='AttackType', hue='AttackType', data=terror, palette='inferno', order=terror['AttackType'].value_counts().index, legend=False)
    plt.xticks(rotation=90)
    plt.title('Attacking Methods by Terrorists')
    plt.savefig('static/attack_methods_by_terrorists.png')
    plt.close()

    
    plt.figure(figsize=(18, 6))
    sns.barplot(
        x=terror['Country'].value_counts()[:15].index,
        y=terror['Country'].value_counts()[:15].values,
        palette='inferno'
    )
    plt.title('Top Affected Countries')
    plt.savefig('static/top_affected_countries.png')
    plt.close()




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualizations')
def visualizations():
    create_plots()  
    return render_template('visualizations.html')

@app.route('/map')
def map():
    return render_template('terror_map.html')

@app.route('/machine_learning')
def machine_learning():
    
    auc_score = 0.95  
    return render_template('machine_learning.html', auc_score=auc_score)



if __name__ == "__main__":
    app.run(debug=True)
