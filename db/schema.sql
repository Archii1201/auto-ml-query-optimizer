-- ==========================================================
-- AutoML-Powered Learned Query Optimizer
-- Phase 1: Schema + Sample Data
-- ==========================================================
-- This file creates the sample tables (customers, orders) and
-- populates them with synthetic rows using generate_series().
-- It also includes a small set of representative queries that
-- are used by collect_data.py to gather EXPLAIN ANALYZE plans.
-- ==========================================================

-- ----- Clean slate (safe to re-run) -----
DROP TABLE IF EXISTS orders    CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ----------------------------------------------------------
-- customers
-- ----------------------------------------------------------
CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    name          TEXT        NOT NULL,
    email         TEXT        NOT NULL,
    country       TEXT        NOT NULL,
    signup_date   DATE        NOT NULL,
    age           INT         NOT NULL
);

-- ----------------------------------------------------------
-- orders
-- ----------------------------------------------------------
CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INT         NOT NULL REFERENCES customers(customer_id),
    order_date    DATE        NOT NULL,
    amount        NUMERIC(10,2) NOT NULL,
    status        TEXT        NOT NULL
);

-- ----------------------------------------------------------
-- Sample data: 10,000 customers
-- ----------------------------------------------------------
INSERT INTO customers (name, email, country, signup_date, age)
SELECT
    'Customer_' || g,
    'user_' || g || '@example.com',
    (ARRAY['USA','India','UK','Germany','Canada','Australia','France','Japan'])[1 + (g % 8)],
    DATE '2018-01-01' + (g % 2000),
    18 + (g % 60)
FROM generate_series(1, 10000) AS g;

-- ----------------------------------------------------------
-- Sample data: 100,000 orders (≈10 orders per customer on avg)
-- ----------------------------------------------------------
INSERT INTO orders (customer_id, order_date, amount, status)
SELECT
    1 + (g % 10000),
    DATE '2022-01-01' + (g % 1000),
    ROUND((random() * 1000)::NUMERIC, 2),
    (ARRAY['PENDING','SHIPPED','DELIVERED','CANCELLED'])[1 + (g % 4)]
FROM generate_series(1, 100000) AS g;

-- ----------------------------------------------------------
-- Indexes (intentionally minimal; lets the planner make choices)
-- ----------------------------------------------------------
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date  ON orders(order_date);

-- Refresh planner statistics
ANALYZE customers;
ANALYZE orders;
