import os

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

load_dotenv()
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}
NEW_DB_NAME = os.getenv("DB_NAME")
NEW_TABLE_NAME = "daily_prices"


def setup_databases():
    """ """
    try:
        print("Connecting")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        print("Successful")
        print(f"Checking if database '{NEW_DB_NAME}' exists...")
        cursor.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), [NEW_DB_NAME]
        )
        if not cursor.fetchone():
            print(f"Database '{NEW_DB_NAME}' not found. Creating it...")
            cursor.execute(sql.SQL(f"CREATE DATABASE {NEW_DB_NAME}"))
            print(f"Database '{NEW_DB_NAME}' created successfully.")
        else:
            print(f"Database '{NEW_DB_NAME}' already exists.")
        cursor.close()
        conn.close()
        print(f"\nConnecting to the new database '{NEW_DB_NAME}'...")
        conn_new = psycopg2.connect(dbname=NEW_DB_NAME, **DB_CONFIG)
        cursor_new = conn_new.cursor()
        print("Connection successful.")

        # Create daily_prices table
        create_prices_table_query = """
                                    CREATE TABLE IF NOT EXISTS daily_prices
                                    (
                                        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                                        symbol    VARCHAR(10)                 NOT NULL,
                                        open      NUMERIC(12, 6),
                                        high      NUMERIC(12, 6),
                                        low       NUMERIC(12, 6),
                                        close     NUMERIC(12, 6),
                                        adj_close NUMERIC(12, 6),
                                        volume    NUMERIC,
                                        PRIMARY KEY (timestamp, symbol)
                                    );
                                    """
        print("Creating table 'daily_prices' if it doesn't exist...")
        cursor_new.execute(create_prices_table_query)
        print("Table 'daily_prices' is ready.")

        # Create macro_data table
        create_macro_table_query = """
                                   CREATE TABLE IF NOT EXISTS macro_data
                                   (
                                       timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                                       series_id VARCHAR(20)                 NOT NULL,
                                       value     NUMERIC(18, 6),
                                       PRIMARY KEY (timestamp, series_id)
                                   );
                                   """
        print("Creating table 'macro_data' if it doesn't exist...")
        cursor_new.execute(create_macro_table_query)
        print("Table 'macro_data' is ready.")

        # Create fundamental_data table
        create_fundamental_table_query = """
                                         CREATE TABLE IF NOT EXISTS fundamental_data
                                         (
                                             timestamp                TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                                             symbol                   VARCHAR(10)                 NOT NULL,

                                             -- Company Profile
                                             company_name             VARCHAR(255),
                                             sector                   VARCHAR(100),
                                             industry                 VARCHAR(100),
                                             market_cap               NUMERIC(20, 2),
                                             price                    NUMERIC(12, 6),
                                             beta                     NUMERIC(8, 4),
                                             dividend_yield           NUMERIC(8, 6),

                                             -- Income Statement
                                             revenue                  NUMERIC(20, 2),
                                             cogs                     NUMERIC(20, 2),
                                             gross_profit             NUMERIC(20, 2),
                                             operating_income         NUMERIC(20, 2),
                                             ebitda                   NUMERIC(20, 2),
                                             earnings                 NUMERIC(20, 2),
                                             eps                      NUMERIC(12, 6),
                                             eps_diluted              NUMERIC(12, 6),

                                             -- Margins
                                             gross_margin             NUMERIC(8, 6),
                                             operating_margin         NUMERIC(8, 6),
                                             profit_margin            NUMERIC(8, 6),

                                             -- Balance Sheet
                                             total_assets             NUMERIC(20, 2),
                                             total_debt               NUMERIC(20, 2),
                                             cash                     NUMERIC(20, 2),
                                             shareholders_equity      NUMERIC(20, 2),
                                             total_liabilities        NUMERIC(20, 2),
                                             current_assets           NUMERIC(20, 2),
                                             current_liabilities      NUMERIC(20, 2),

                                             -- Cash Flow Statement
                                             operating_cash_flow      NUMERIC(20, 2),
                                             free_cash_flow           NUMERIC(20, 2),
                                             capex                    NUMERIC(20, 2),
                                             dividends_paid           NUMERIC(20, 2),

                                             -- Key Metrics
                                             pe_ratio                 NUMERIC(12, 6),
                                             price_to_book            NUMERIC(12, 6),
                                             roe                      NUMERIC(8, 6),
                                             roa                      NUMERIC(8, 6),
                                             revenue_per_share        NUMERIC(12, 6),
                                             free_cash_flow_per_share NUMERIC(12, 6),
                                             book_value               NUMERIC(12, 6),
                                             tangible_book_value      NUMERIC(12, 6),
                                             debt_to_equity           NUMERIC(12, 6),
                                             current_ratio            NUMERIC(8, 6),
                                             quick_ratio              NUMERIC(8, 6),

                                             -- Ratios
                                             pe_ratio_ttm             NUMERIC(12, 6),
                                             price_to_sales           NUMERIC(12, 6),
                                             price_to_book_ttm        NUMERIC(12, 6),
                                             price_to_fcf             NUMERIC(12, 6),
                                             ev_to_ebitda             NUMERIC(12, 6),
                                             peg_ratio                NUMERIC(12, 6),
                                             debt_ratio               NUMERIC(8, 6),
                                             asset_turnover           NUMERIC(8, 6),
                                             inventory_turnover       NUMERIC(8, 6),
                                             receivables_turnover     NUMERIC(8, 6),
                                             dividend_payout_ratio    NUMERIC(8, 6),

                                             PRIMARY KEY (timestamp, symbol)
                                         );
                                         """
        print("Creating table 'fundamental_data' if it doesn't exist...")
        cursor_new.execute(create_fundamental_table_query)
        print("Table 'fundamental_data' is ready.")

        conn_new.commit()
        cursor_new.close()
        conn_new.close()
        print("\nDatabase setup is complete!")
    except psycopg2.OperationalError as e:
        print(f"\n--- CONNECTION ERROR ---: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    setup_databases()
