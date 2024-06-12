from flask import Flask
app=Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the Home Page'
@app.route('/hello_world')
def hello_world():
    return 'Hello Vedanti'
if __name__=='__main__':
    app.run(debug=True)