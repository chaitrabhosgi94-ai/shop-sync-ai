from flask import Flask, render_template, request, redirect, url_for

import sqlite3
import json

from datetime import datetime
from pathlib import Path

from ml.predictor import predict_demand


 
BASE_DIR = Path(__file__).resolve().parent

DATABASE = BASE_DIR / "database.db"

METRICS_PATH = BASE_DIR / "ml" / "model_metrics.json"

 
app = Flask(__name__)


 
def get_db_connection():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


 
def create_database():

    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            category TEXT NOT NULL,

            price REAL NOT NULL,

            stock INTEGER NOT NULL

        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            product TEXT NOT NULL,

            quantity INTEGER NOT NULL,

            sale_date TEXT NOT NULL

        )
    """)

    product_count = conn.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    if product_count == 0:

        products = [

            ("Rice", "Grocery", 450, 15),

            ("Milk", "Dairy", 60, 20),

            ("Bread", "Bakery", 45, 10),

            ("Biscuits", "Snacks", 30, 25),

            ("Soap", "Personal Care", 40, 12)

        ]

        conn.executemany("""
            INSERT INTO products
            (name, category, price, stock)

            VALUES (?, ?, ?, ?)
        """, products)

    conn.commit()

    conn.close()

 
@app.route("/")
def index():

    conn = get_db_connection()

    products = conn.execute(
        "SELECT * FROM products"
    ).fetchall()

    total_products = conn.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    total_stock = conn.execute(
        "SELECT SUM(stock) FROM products"
    ).fetchone()[0] or 0

    total_sales = conn.execute(
        "SELECT SUM(quantity) FROM sales"
    ).fetchone()[0] or 0

    low_stock = conn.execute("""
        SELECT COUNT(*)
        FROM products
        WHERE stock < 10
    """).fetchone()[0]

    sales_rows = conn.execute("""
        SELECT sale_date, SUM(quantity) AS total
        FROM sales
        GROUP BY sale_date
        ORDER BY sale_date
    """).fetchall()

    sales_dates = [
        row["sale_date"]
        for row in sales_rows
    ]

    sales_values = [
        row["total"]
        for row in sales_rows
    ]

    product_names = [
        product["name"]
        for product in products
    ]

    product_stocks = [
        product["stock"]
        for product in products
    ]

    category_rows = conn.execute("""
        SELECT category, SUM(stock) AS total
        FROM products
        GROUP BY category
    """).fetchall()

    category_names = [
        row["category"]
        for row in category_rows
    ]

    category_values = [
        row["total"]
        for row in category_rows
    ]

    ai_products = []

    ai_demands = []

    for product in products:

        try:

            demand = predict_demand(
                product["name"]
            )

        except Exception as error:

            print(
                "AI prediction error:",
                error
            )

            demand = None

        if demand is not None:

            ai_products.append(
                product["name"]
            )

            ai_demands.append(
                demand
            )

    conn.close()

    return render_template(
        "index.html",

        products=products,

        total_products=total_products,

        total_stock=total_stock,

        total_sales=total_sales,

        low_stock=low_stock,

        sales_dates=sales_dates,

        sales_values=sales_values,

        product_names=product_names,

        product_stocks=product_stocks,

        category_names=category_names,

        category_values=category_values,

        ai_products=ai_products,

        ai_demands=ai_demands
    )

 
@app.route("/inventory")
def inventory():

    conn = get_db_connection()

    products = conn.execute(
        "SELECT * FROM products"
    ).fetchall()

    conn.close()

    return render_template(
        "inventory.html",
        products=products
    )


 
@app.route(
    "/add_product",
    methods=["POST"]
)
def add_product():

    name = request.form["name"]

    category = request.form["category"]

    price = float(
        request.form["price"]
    )

    stock = int(
        request.form["stock"]
    )

    conn = get_db_connection()

    conn.execute("""
        INSERT INTO products
        (name, category, price, stock)

        VALUES (?, ?, ?, ?)
    """, (
        name,
        category,
        price,
        stock
    ))

    conn.commit()

    conn.close()

    return redirect(
        url_for("inventory")
    )


 
@app.route(
    "/delete_product/<int:product_id>"
)
def delete_product(product_id):

    conn = get_db_connection()

    conn.execute(
        "DELETE FROM products WHERE id = ?",
        (product_id,)
    )

    conn.commit()

    conn.close()

    return redirect(
        url_for("inventory")
    )


 
@app.route("/sales")
def sales():

    conn = get_db_connection()

    sales_data = conn.execute("""
        SELECT *
        FROM sales
        ORDER BY id DESC
    """).fetchall()

    products = conn.execute("""
        SELECT *
        FROM products
    """).fetchall()

    total_sales = conn.execute("""
        SELECT SUM(quantity)
        FROM sales
    """).fetchone()[0] or 0

    total_revenue = 0

    for sale in sales_data:

        product = conn.execute("""
            SELECT price
            FROM products
            WHERE name = ?
        """, (
            sale["product"],
        )).fetchone()

        if product:

            total_revenue += (
                product["price"]
                *
                sale["quantity"]
            )

    transaction_count = len(
        sales_data
    )

    conn.close()

    return render_template(
        "sales.html",

        sales=sales_data,

        products=products,

        total_sales=total_sales,

        total_revenue=total_revenue,

        transaction_count=transaction_count
    )


 
@app.route(
    "/add_sale",
    methods=["POST"]
)
def add_sale():

    product = request.form["product"]

    quantity = int(
        request.form["quantity"]
    )

    if quantity <= 0:

        return """
        <h2>Invalid Quantity</h2>

        <a href="/sales">
        Back to Sales
        </a>
        """

    sale_date = datetime.now().strftime(
        "%Y-%m-%d"
    )

    conn = get_db_connection()

    product_data = conn.execute("""
        SELECT stock
        FROM products
        WHERE name = ?
    """, (
        product,
    )).fetchone()

    if product_data is None:

        conn.close()

        return """
        <h2>Product Not Found</h2>

        <a href="/sales">
        Back to Sales
        </a>
        """

    current_stock = product_data["stock"]

    if quantity > current_stock:

        conn.close()

        return f"""
        <h2>Not Enough Stock</h2>

        <p>
        Available stock: {current_stock}
        </p>

        <a href="/sales">
        Back to Sales
        </a>
        """

    conn.execute("""
        INSERT INTO sales
        (product, quantity, sale_date)

        VALUES (?, ?, ?)
    """, (
        product,
        quantity,
        sale_date
    ))

    conn.execute("""
        UPDATE products

        SET stock = stock - ?

        WHERE name = ?
    """, (
        quantity,
        product
    ))

    conn.commit()

    conn.close()

    return redirect(
        url_for("sales")
    )


 
def load_metrics():

    metrics = {

        "model":
            "Random Forest Regression",

        "mae":
            "N/A",

        "r2_score":
            "N/A",

        "training_records":
            "N/A",

        "testing_records":
            "N/A"

    }

    if METRICS_PATH.exists():

        try:

            with open(
                METRICS_PATH,
                "r"
            ) as file:

                metrics = json.load(
                    file
                )

        except Exception as error:

            print(
                "Metrics loading error:",
                error
            )

    return metrics


 
@app.route("/ai-forecast")
def ai_forecast():

    conn = get_db_connection()

    products = conn.execute(
        "SELECT * FROM products"
    ).fetchall()

    conn.close()

    forecasts = []

    for product in products:

        try:

            demand = predict_demand(
                product["name"]
            )

        except Exception as error:

            print(
                "Forecast error:",
                error
            )

            demand = None

        if demand is None:

            continue

        reorder_quantity = max(
            0,
            demand - product["stock"]
        )

        if demand >= 40:

            demand_level = "High"

        elif demand >= 20:

            demand_level = "Medium"

        else:

            demand_level = "Low"

        if reorder_quantity > 0:

            recommendation = (
                f"Order {reorder_quantity} "
                f"additional units."
            )

        else:

            recommendation = (
                "Current stock is sufficient."
            )

        forecasts.append({

            "product":
                product["name"],

            "category":
                product["category"],

            "predicted_demand":
                demand,

            "current_stock":
                product["stock"],

            "reorder_quantity":
                reorder_quantity,

            "demand_level":
                demand_level,

            "recommendation":
                recommendation

        })

    metrics = load_metrics()

    return render_template(
        "ai_forecast.html",

        forecasts=forecasts,

        metrics=metrics
    )



@app.route("/forecast")
def forecast():

    return redirect(
        url_for("ai_forecast")
    )


 
@app.route("/insights")
def insights():

    conn = get_db_connection()

    products = conn.execute(
        "SELECT * FROM products"
    ).fetchall()

    total_products = conn.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]

    total_stock = conn.execute(
        "SELECT SUM(stock) FROM products"
    ).fetchone()[0] or 0

    total_sales = conn.execute(
        "SELECT SUM(quantity) FROM sales"
    ).fetchone()[0] or 0

    low_stock_products = conn.execute("""
        SELECT *
        FROM products
        WHERE stock < 10
        ORDER BY stock ASC
    """).fetchall()

    conn.close()

    insights_data = []

    for product in products:

        try:

            demand = predict_demand(
                product["name"]
            )

        except Exception as error:

            print(
                "Insight prediction error:",
                error
            )

            demand = None

        if demand is None:

            continue

        current_stock = product["stock"]

        reorder = max(
            0,
            demand - current_stock
        )

        if demand >= 40:

            demand_level = "High"

            status = (
                "Strong demand expected"
            )

        elif demand >= 20:

            demand_level = "Medium"

            status = (
                "Moderate demand expected"
            )

        else:

            demand_level = "Low"

            status = (
                "Lower demand expected"
            )

        if reorder > 0:

            recommendation = (
                f"Consider ordering "
                f"{reorder} units."
            )

        else:

            recommendation = (
                "Current stock is sufficient."
            )

        insights_data.append({

            "product":
                product["name"],

            "category":
                product["category"],

            "stock":
                current_stock,

            "demand":
                demand,

            "reorder":
                reorder,

            "demand_level":
                demand_level,

            "status":
                status,

            "recommendation":
                recommendation

        })

    highest_demand = None

    if insights_data:

        highest_demand = max(
            insights_data,
            key=lambda item:
                item["demand"]
        )

    return render_template(
        "insights.html",

        insights=insights_data,

        total_products=
            total_products,

        total_stock=
            total_stock,

        total_sales=
            total_sales,

        low_stock_products=
            low_stock_products,

        highest_demand=
            highest_demand
    )


 
if __name__ == "__main__":

    create_database()

    print("")
    print("====================================")
    print("        SHOP SYNC AI")
    print("====================================")
    print("")
    print("Dashboard:")
    print("http://127.0.0.1:5000/")
    print("")
    print("Inventory:")
    print("http://127.0.0.1:5000/inventory")
    print("")
    print("Sales:")
    print("http://127.0.0.1:5000/sales")
    print("")
    print("AI Forecast:")
    print("http://127.0.0.1:5000/ai-forecast")
    print("")
    print("AI Insights:")
    print("http://127.0.0.1:5000/insights")
    print("")
    print("====================================")

    app.run(
        debug=True
    )