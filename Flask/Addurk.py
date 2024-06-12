from flask import Flask, request, redirect
app=Flask(__name__)
@app.route('/')
def index():
    return 'Welcome to the Home Page'
@app.route('/hello')
def hello():
    return 'Hello Vedanti'
def home():
    return 'My Home Page'
app.add_url_rule('/home','home',home)
if __name__=='__main__':
    app.run(debug=True)