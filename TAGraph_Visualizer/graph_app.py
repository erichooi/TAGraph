import json
from flask import Flask, render_template, request, jsonify, redirect, abort

app = Flask(__name__, static_url_path="/static")

json_data = None

@app.route("/")
def index():
	return render_template("visualize_data.html")

@app.route("/data", methods=["GET"])
def get_json_data():
	if request.method == "GET":
		global json_data
		if json_data == None:
			return abort(404)
		else:
			return jsonify(json.loads(json_data))

@app.route("/update_graph", methods=["POST"])
def post_graph_data():
	if request.method == "POST":
		try:
			global json_data
			f = request.files["graph_file"]
			json_data = f.read()
			return redirect("/")
		except:
			return redirect("/")

if __name__ == "__main__":
	app.run()