-- ==========================================================
-- AutoML-Powered Learned Query Optimizer
-- Phase 2A: TPC-H Benchmark Schema (PostgreSQL flavor)
-- ==========================================================
-- Standard TPC-H decision-support schema:
--   region, nation, supplier, customer, part, partsupp, orders, lineitem
--
-- Reference: TPC-H Standard Specification (Section 1.4)
-- ----------------------------------------------------------
-- This file is safe to re-run: it drops existing TPC-H tables
-- before recreating them. Indexes are intentionally minimal so
-- the optimizer has room to make varied plan choices.
-- ==========================================================

-- ----- Clean slate (child-first because of FK chain) -----
DROP TABLE IF EXISTS lineitem CASCADE;
DROP TABLE IF EXISTS orders   CASCADE;
DROP TABLE IF EXISTS partsupp CASCADE;
DROP TABLE IF EXISTS customer CASCADE;
DROP TABLE IF EXISTS supplier CASCADE;
DROP TABLE IF EXISTS part     CASCADE;
DROP TABLE IF EXISTS nation   CASCADE;
DROP TABLE IF EXISTS region   CASCADE;

-- ----------------------------------------------------------
-- region  (5 rows at any SF)
-- ----------------------------------------------------------
CREATE TABLE region (
    r_regionkey  INTEGER        NOT NULL,
    r_name       CHAR(25)       NOT NULL,
    r_comment    VARCHAR(152),
    PRIMARY KEY (r_regionkey)
);

-- ----------------------------------------------------------
-- nation  (25 rows at any SF)
-- ----------------------------------------------------------
CREATE TABLE nation (
    n_nationkey  INTEGER        NOT NULL,
    n_name       CHAR(25)       NOT NULL,
    n_regionkey  INTEGER        NOT NULL,
    n_comment    VARCHAR(152),
    PRIMARY KEY (n_nationkey),
    FOREIGN KEY (n_regionkey) REFERENCES region(r_regionkey)
);

-- ----------------------------------------------------------
-- part  (200,000 rows at SF1)
-- ----------------------------------------------------------
CREATE TABLE part (
    p_partkey     INTEGER        NOT NULL,
    p_name        VARCHAR(55)    NOT NULL,
    p_mfgr        CHAR(25)       NOT NULL,
    p_brand       CHAR(10)       NOT NULL,
    p_type        VARCHAR(25)    NOT NULL,
    p_size        INTEGER        NOT NULL,
    p_container   CHAR(10)       NOT NULL,
    p_retailprice DECIMAL(15,2)  NOT NULL,
    p_comment     VARCHAR(23)    NOT NULL,
    PRIMARY KEY (p_partkey)
);

-- ----------------------------------------------------------
-- supplier  (10,000 rows at SF1)
-- ----------------------------------------------------------
CREATE TABLE supplier (
    s_suppkey   INTEGER        NOT NULL,
    s_name      CHAR(25)       NOT NULL,
    s_address   VARCHAR(40)    NOT NULL,
    s_nationkey INTEGER        NOT NULL,
    s_phone     CHAR(15)       NOT NULL,
    s_acctbal   DECIMAL(15,2)  NOT NULL,
    s_comment   VARCHAR(101)   NOT NULL,
    PRIMARY KEY (s_suppkey),
    FOREIGN KEY (s_nationkey) REFERENCES nation(n_nationkey)
);

-- ----------------------------------------------------------
-- partsupp  (800,000 rows at SF1)
-- ----------------------------------------------------------
CREATE TABLE partsupp (
    ps_partkey    INTEGER        NOT NULL,
    ps_suppkey    INTEGER        NOT NULL,
    ps_availqty   INTEGER        NOT NULL,
    ps_supplycost DECIMAL(15,2)  NOT NULL,
    ps_comment    VARCHAR(199)   NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey),
    FOREIGN KEY (ps_partkey) REFERENCES part(p_partkey),
    FOREIGN KEY (ps_suppkey) REFERENCES supplier(s_suppkey)
);

-- ----------------------------------------------------------
-- customer  (150,000 rows at SF1)
-- ----------------------------------------------------------
CREATE TABLE customer (
    c_custkey    INTEGER        NOT NULL,
    c_name       VARCHAR(25)    NOT NULL,
    c_address    VARCHAR(40)    NOT NULL,
    c_nationkey  INTEGER        NOT NULL,
    c_phone      CHAR(15)       NOT NULL,
    c_acctbal    DECIMAL(15,2)  NOT NULL,
    c_mktsegment CHAR(10)       NOT NULL,
    c_comment    VARCHAR(117)   NOT NULL,
    PRIMARY KEY (c_custkey),
    FOREIGN KEY (c_nationkey) REFERENCES nation(n_nationkey)
);

-- ----------------------------------------------------------
-- orders  (1,500,000 rows at SF1)
-- ----------------------------------------------------------
CREATE TABLE orders (
    o_orderkey      INTEGER        NOT NULL,
    o_custkey       INTEGER        NOT NULL,
    o_orderstatus   CHAR(1)        NOT NULL,
    o_totalprice    DECIMAL(15,2)  NOT NULL,
    o_orderdate     DATE           NOT NULL,
    o_orderpriority CHAR(15)       NOT NULL,
    o_clerk         CHAR(15)       NOT NULL,
    o_shippriority  INTEGER        NOT NULL,
    o_comment       VARCHAR(79)    NOT NULL,
    PRIMARY KEY (o_orderkey),
    FOREIGN KEY (o_custkey) REFERENCES customer(c_custkey)
);

-- ----------------------------------------------------------
-- lineitem  (~6,000,000 rows at SF1)
-- ----------------------------------------------------------
CREATE TABLE lineitem (
    l_orderkey      INTEGER        NOT NULL,
    l_partkey       INTEGER        NOT NULL,
    l_suppkey       INTEGER        NOT NULL,
    l_linenumber    INTEGER        NOT NULL,
    l_quantity      DECIMAL(15,2)  NOT NULL,
    l_extendedprice DECIMAL(15,2)  NOT NULL,
    l_discount      DECIMAL(15,2)  NOT NULL,
    l_tax           DECIMAL(15,2)  NOT NULL,
    l_returnflag    CHAR(1)        NOT NULL,
    l_linestatus    CHAR(1)        NOT NULL,
    l_shipdate      DATE           NOT NULL,
    l_commitdate    DATE           NOT NULL,
    l_receiptdate   DATE           NOT NULL,
    l_shipinstruct  CHAR(25)       NOT NULL,
    l_shipmode      CHAR(10)       NOT NULL,
    l_comment       VARCHAR(44)    NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber),
    FOREIGN KEY (l_orderkey)            REFERENCES orders(o_orderkey),
    FOREIGN KEY (l_partkey, l_suppkey)  REFERENCES partsupp(ps_partkey, ps_suppkey)
);

-- ----------------------------------------------------------
-- Reference indexes (lightweight; lets the planner make choices).
-- These match the indexes commonly used in TPC-H research papers
-- so our plans are directly comparable to published baselines.
-- ----------------------------------------------------------
CREATE INDEX idx_lineitem_shipdate     ON lineitem(l_shipdate);
CREATE INDEX idx_lineitem_orderkey     ON lineitem(l_orderkey);
CREATE INDEX idx_lineitem_partkey      ON lineitem(l_partkey);
CREATE INDEX idx_lineitem_suppkey      ON lineitem(l_suppkey);
CREATE INDEX idx_lineitem_part_supp    ON lineitem(l_partkey, l_suppkey);
CREATE INDEX idx_orders_orderdate      ON orders(o_orderdate);
CREATE INDEX idx_orders_custkey        ON orders(o_custkey);
CREATE INDEX idx_customer_nationkey    ON customer(c_nationkey);
CREATE INDEX idx_supplier_nationkey    ON supplier(s_nationkey);
CREATE INDEX idx_partsupp_suppkey      ON partsupp(ps_suppkey);
CREATE INDEX idx_nation_regionkey      ON nation(n_regionkey);
