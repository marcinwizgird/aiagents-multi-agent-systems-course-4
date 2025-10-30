import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import smolagents as smol
# Correct imports: Agent is needed for the orchestrator
from smolagents import OpenAIServerModel, tool
from pydantic import BaseModel, Field  # Import Pydantic
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
from sqlalchemy import create_engine, Engine

SEMANTIC_SEARCH_ENABLED = True

# --- Semantic Search Imports ---
# Attempt to import necessary libraries for semantic search.
# If they are not available, semantic search will be disabled.
try:
    # We still use sklearn for cosine_similarity, even with OpenAI embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    # Check if openai library is available
    import openai
    SEMANTIC_SEARCH_ENABLED = True
    print("OpenAI and scikit-learn libraries found. Enabling semantic search.")
except ImportError:
    print("WARNING: 'openai' or 'scikit-learn' not found.")
    print("Semantic search feature will be disabled.")
    print("To enable it, run: pip install openai scikit-learn")
    SEMANTIC_SEARCH_ENABLED = False
    cosine_similarity = None
    openai = None
# --- End Semantic Search Imports ---


# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates", "category": "product", "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins", "category": "product", "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups", "category": "product", "unit_price": 0.10},  # per cup
    {"item_name": "Table covers", "category": "product", "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes", "category": "product", "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes", "category": "product", "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards", "category": "product", "unit_price": 0.50},  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers", "category": "product", "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags", "category": "product", "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards", "category": "product", "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders", "category": "product", "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]


# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.
    """
    # ... (function body unchanged) ...
    np.random.seed(seed)
    num_items = int(len(paper_supplies) * coverage)
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )
    selected_items = [paper_supplies[i] for i in selected_indices]
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),
            "min_stock_level": np.random.randint(50, 150)
        })
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.
    """
    # ... (function body unchanged) ...
    try:
        transactions_schema = pd.DataFrame({
            "id": [], "item_name": [], "transaction_type": [],
            "units": [], "price": [], "transaction_date": [],
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)
        initial_date = datetime(2025, 1, 1).isoformat()
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))
        quotes_df = quotes_df[[
            "request_id", "total_amount", "quote_explanation",
            "order_date", "job_type", "order_size", "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)
        initial_transactions = []
        initial_transactions.append({
            "item_name": None, "transaction_type": "sales",
            "units": None, "price": 50000.0, "transaction_date": initial_date,
        })
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"], "transaction_type": "stock_orders",
                "units": item["current_stock"], "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)
        return db_engine
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def create_transaction(
        item_name: str,
        transaction_type: str,
        quantity: int,
        price: float,
        date: Union[str, datetime],
) -> int:
    """
    Records a single transaction into the 'transactions' table.
    """
    # ... (function body unchanged) ...
    try:
        date_str = date.isoformat() if isinstance(date, datetime) else date
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")
        transaction = pd.DataFrame([{
            "item_name": item_name, "transaction_type": transaction_type,
            "units": quantity, "price": price, "transaction_date": date_str,
        }])
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])
    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of all available inventory as of a specific date.
    """
    # ... (function body unchanged) ...
    query = """
            SELECT item_name, \
                   SUM(CASE \
                           WHEN transaction_type = 'stock_orders' THEN units \
                           WHEN transaction_type = 'sales' THEN -units \
                           ELSE 0 \
                       END) as stock
            FROM transactions
            WHERE item_name IS NOT NULL
              AND transaction_date <= :as_of_date
            GROUP BY item_name
            HAVING stock > 0 \
            """
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.
    """
    # ... (function body unchanged) ...
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()
    stock_query = """
                  SELECT item_name, \
                         COALESCE(SUM(CASE \
                                          WHEN transaction_type = 'stock_orders' THEN units \
                                          WHEN transaction_type = 'sales' THEN -units \
                                          ELSE 0 \
                             END), 0) AS current_stock
                  FROM transactions
                  WHERE item_name = :item_name
                    AND transaction_date <= :as_of_date \
                  """
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity.
    """
    # ... (function body unchanged) ...
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7
    delivery_date_dt = input_date_dt + timedelta(days=days)
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.
    """
    # ... (function body unchanged) ...
    try:
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)
        return 0.0
    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.
    """
    # ... (function body unchanged) ...
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()
    cash = get_cash_balance(as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value
        inventory_summary.append({
            "item_name": item["item_name"], "stock": stock,
            "unit_price": item["unit_price"], "value": item_value,
        })
    top_sales_query = """
                      SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
                      FROM transactions
                      WHERE transaction_type = 'sales' \
                        AND transaction_date <= :date
                      GROUP BY item_name \
                      ORDER BY total_revenue DESC LIMIT 5 \
                      """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    return {
        "as_of_date": as_of_date, "cash_balance": cash,
        "inventory_value": inventory_value, "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_sales.to_dict(orient="records"),
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.
    """
    # ... (function body unchanged) ...
    conditions = []
    params = {}
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT
            qr.response AS original_request, q.total_amount, q.quote_explanation,
            q.job_type, q.order_size, q.event_type, q.order_date
        FROM quotes q JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause} ORDER BY q.order_date DESC LIMIT {limit}
    """
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row) for row in result]


def get_item_details(item_name: str) -> pd.DataFrame:
    """
    Retrieve all details for a specific item from the 'inventory' table.
    """
    # ... (function body unchanged) ...
    try:
        query = "SELECT item_name, category, unit_price, min_stock_level FROM inventory WHERE item_name = :item_name"
        return pd.read_sql(query, db_engine, params={"item_name": item_name})
    except Exception as e:
        print(f"Error getting item details for {item_name}: {e}")
        return pd.DataFrame()


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv()
api_key = os.getenv("UDACITY_OPENAI_API_KEY")

if not api_key:
    raise ValueError("UDACITY_OPENAI_API_KEY not found in .env file")

api_base = os.getenv("UDACITY_OPENAI_API_BASE", "https://openai.vocareum.com/v1") # Default if not set

# Configure the model explicitly
model = OpenAIServerModel(
    model_id="gpt-4o-mini",  # Use a modern, capable model
    api_key=api_key,
    api_base="https://openai.vocareum.com/v1",
)

# --- OpenAI Client Setup (for embeddings) ---
openai_client = None
if SEMANTIC_SEARCH_ENABLED:
    try:
        openai_client = openai.OpenAI(api_key=api_key, base_url=api_base)
        print("OpenAI client initialized for embeddings.")
    except Exception as e:
        print(f"ERROR initializing OpenAI client: {e}")
        SEMANTIC_SEARCH_ENABLED = False
# --- End OpenAI Client Setup ---

# --- Precompute Embeddings ---
item_embedding_map = {} # {item_name: numpy_embedding_vector}

def precompute_openai_embeddings(client, items_list):
    """Calculates and stores embeddings for item names using OpenAI API."""
    global item_embedding_map # Ensure we modify the global map
    global SEMANTIC_SEARCH_ENABLED
    if not client or not SEMANTIC_SEARCH_ENABLED:
        print("Skipping embedding precomputation (disabled or client failed).")
        return {} # Return empty map

    item_names = [item['item_name'] for item in items_list]
    if not item_names:
        print("No item names found in paper_supplies to embed.")
        return {}

    try:
        print(f"Requesting OpenAI embeddings for {len(item_names)} items...")
        # Use a recommended embedding model (check OpenAI docs for latest)
        response = client.embeddings.create(
            input=item_names,
            model="text-embedding-3-small" # Or another suitable model like text-embedding-ada-002
        )

        # Check if response structure is as expected
        if response.data and len(response.data) == len(item_names):
             item_embeddings_raw = [item.embedding for item in response.data]
             # Store as numpy arrays for easier similarity calculation
             temp_map = {name: np.array(emb) for name, emb in zip(item_names, item_embeddings_raw)}
             print(f"Successfully precomputed embeddings for {len(temp_map)} items.")
             return temp_map
        else:
             print("ERROR: Unexpected response structure from OpenAI embeddings API.")
             print(f"Response: {response}") # Log the response for debugging
             raise ValueError("Embeddings data not found or mismatch in count.")

    except Exception as e:
        print(f"ERROR during OpenAI embedding precomputation: {e}")
        print("Disabling semantic search due to precomputation failure.")
        #global SEMANTIC_SEARCH_ENABLED # Need to modify global flag
        SEMANTIC_SEARCH_ENABLED = False
        return {} # Return empty map on failure


# --- Call precomputation function once at startup ---
item_embedding_map = precompute_openai_embeddings(openai_client, paper_supplies)
# --- End Precomputation ---

# --- Pydantic Models for Structured I/O ---

class HistoricalQuote(BaseModel):
    original_request: str
    total_amount: float
    quote_explanation: str
    order_date: str


class QuoteHistoryResponse(BaseModel):
    matches_found: int
    matches: List[HistoricalQuote]


class TransactionConfirmation(BaseModel):
    transaction_id: int
    item_name: str
    quantity: int
    total_price: float
    status: str = "Successfully created sale transaction"


class TopSeller(BaseModel):
    item_name: str
    total_units: int
    total_revenue: float


class InventoryItemSummary(BaseModel):
    item_name: str
    stock: int
    unit_price: float
    value: float


class FinancialReport(BaseModel):
    as_of_date: str
    cash_balance: float
    inventory_value: float
    total_assets: float
    inventory_summary: List[InventoryItemSummary]
    top_selling_products: List[TopSeller]


class InventoryStockReport(BaseModel):
    as_of_date: str
    stock_levels: Dict[str, int]


class InventoryValidationReport(BaseModel):
    status: str = "VALIDATION_SUCCESS"
    item_name: str
    requested_quantity: int
    unit_price: float
    inventory_item_summary : InventoryItemSummary
    delivery_note: str
    restock_note: str


class InventoryValidationFailure(BaseModel):
    status: str = "VALIDATION_FAILED"
    item_name: str
    reason: str


InventoryValidationResult = Union[InventoryValidationReport, InventoryValidationFailure]

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# --- Deterministic Mapping for Item Names ---
ITEM_ALIAS_MAP = {
    "a4 paper": "A4 paper",
    "letter paper": "Letter-sized paper",
    "letter-size paper": "Letter-sized paper",
    "card stock": "Cardstock",
    "colored paper": "Colored paper",
    "colourful cardstock": "Colored paper",
    "bright-colored paper": "Bright-colored paper",
    "copy paper": "Standard copy paper",
    "standard printer paper": "Standard copy paper",
    "printer paper": "Standard copy paper",
    "plates": "Paper plates",
    "cups": "Paper cups",
    "paper cup": "Paper cups",
    "disposable cups": "Disposable cups",
    "napkins": "Paper napkins",
    "envelopes": "Envelopes"
}


# --- Tool for Quoting Agent ---

@tool
def tool_search_quote_history(search_terms: List[str]) -> QuoteHistoryResponse:
    """
    Searches historical quotes for similar requests to inform new pricing.

    Args:
        search_terms (List[str]): A list of keywords from the customer's request
                                  (e.g., item names, event type).

    Returns:
        QuoteHistoryResponse: A Pydantic model containing a list of matches and a count.
    """
    try:
        results = search_quote_history(search_terms, limit=3)
        return QuoteHistoryResponse(matches_found=len(results), matches=results)
    except Exception as e:
        return QuoteHistoryResponse(matches_found=0, matches=[])


# --- Tool for Order Processor Agent ---

@tool
def tool_create_sale_transaction(item_name: str, quantity: int, total_price: float,
                                 as_of_date: str) -> TransactionConfirmation:
    """
    Finalizes a sale and records it in the database. This *reduces* stock.
    This is ONLY called after validation is complete.

    Args:
        item_name (str): The exact name of the item sold.
        quantity (int): The number of units sold.
        total_price (float): The total sale amount (must be positive).
        as_of_date (str): The date of the sale (ISO format YYYY-MM-DD).

    Returns:
        TransactionConfirmation: A Pydantic model confirming the transaction or
                                 capturing the error.
    """
    try:
        if total_price <= 0:
            raise ValueError("Sale price must be positive.")

        tx_id = create_transaction(item_name, "sales", quantity, total_price, as_of_date)
        return TransactionConfirmation(
            transaction_id=tx_id,
            item_name=item_name,
            quantity=quantity,
            total_price=total_price
        )
    except Exception as e:
        return TransactionConfirmation(
            transaction_id=-1,
            item_name=item_name,
            quantity=quantity,
            total_price=total_price,
            status=f"Error creating sale transaction: {e}"
        )


# --- Semantic Search Tool ---
@tool
def tool_find_closest_item(query_term: str) -> str:
    """
    Finds the item_name from the paper_supplies list that is semantically
    closest to the user's query_term using OpenAI embedding similarity.
    Falls back to a simple alias map if semantic search is disabled or fails.

    Args:
        query_term (str): The item description provided by the user
                          (e.g., "printer paper", "colourful cardstock").

    Returns:
        str: The best matching exact item_name from the paper_supplies list,
             or an error message starting with "ERROR:" if no good match is found.
    """
    # --- Fallback Logic ---
    def fallback_search(term):
        mapped_name = ITEM_ALIAS_MAP.get(term.lower())
        if mapped_name:
             print(f"Semantic Search Disabled/Failed: Using alias map for '{term}' -> '{mapped_name}'")
             return mapped_name
        else:
             print(f"Semantic Search Disabled/Failed: No alias found for '{term}', returning original.")
             # Return original term - validation tool will likely fail, which is correct.
             return term

    if not SEMANTIC_SEARCH_ENABLED or not item_embedding_map or not openai_client:
        return fallback_search(query_term)
    # --- End Fallback Logic ---

    if not query_term:
        return "ERROR: No query term provided."

    try:
        # 1. Get embedding for the query term
        response = openai_client.embeddings.create(
            input=[query_term],
             model="text-embedding-3-small" # Use the same model as precomputation
        )
        if not response.data:
            raise ValueError("No embedding data returned from OpenAI API.")

        query_embedding = np.array(response.data[0].embedding)

        # 2. Prepare precomputed embeddings and names
        precomputed_names = list(item_embedding_map.keys())
        precomputed_embeddings = np.array(list(item_embedding_map.values()))

        # 3. Calculate cosine similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            precomputed_embeddings
        )[0]

        # 4. Find the best match
        best_match_index = np.argmax(similarities)
        best_score = similarities[best_match_index]

        # 5. Apply threshold
        threshold = 0.7 # Adjust this threshold based on testing
        best_matching_item_name = precomputed_names[best_match_index]

        if best_score < threshold:
             # If score is low, try the alias map as a secondary check
             alias_match = ITEM_ALIAS_MAP.get(query_term.lower())
             if alias_match and alias_match == best_matching_item_name:
                 print(f"Semantic Search: Low score {best_score:.2f} for '{query_term}' -> '{best_matching_item_name}', but confirmed by alias map.")
                 return best_matching_item_name
             elif alias_match:
                 print(f"Semantic Search: Low score {best_score:.2f} for '{query_term}'. Falling back to alias map -> '{alias_match}'.")
                 return alias_match
             else:
                  # Return the low-scoring match anyway, let validation decide? Or return error?
                  # Let's return error for now if alias doesn't match either.
                 print(f"Semantic Search: Score {best_score:.2f} < {threshold} for '{query_term}'. No alias match. Failing.")
                 return f"ERROR: Could not find a close match for '{query_term}'. Best score {best_score:.2f} is below threshold {threshold}."
        else:
            print(f"Semantic Search: '{query_term}' -> '{best_matching_item_name}' (Score: {best_score:.2f})")
            return best_matching_item_name

    except Exception as e:
        print(f"ERROR during OpenAI semantic search for '{query_term}': {e}")
        # Fallback to alias map on error
        return fallback_search(query_term)

# --- NEW DETERMINISTIC "SUPER-TOOL" FOR VALIDATION ---
@tool
def validateInventoryItemTool(item_query_term: str, requested_quantity: int, as_of_date: str) -> InventoryValidationResult:
    """
    Processes a single item from an order request.
    1. Finds the exact item name using semantic search (tool_find_closest_item).
    2. Performs inventory validation (stock check, supplier check if needed).
    3. Triggers auto-restock based on minimum levels and cash availability.
    This tool is fully deterministic and returns a structured Pydantic model.

    Args:
        item_query_term (str): The item description provided by the user (e.g., "copy paper").
        requested_quantity (int): The number of units the customer wants.
        as_of_date (str): The date of the request (ISO format YYYY-MM-DD).

    Returns:
        ValidationResult: A Pydantic model (`ValidationSuccess` or `ValidationFailure`).
    """
    # --- Step 1: Find Exact Item Name ---
    total_cost = 0
    exact_item_name = tool_find_closest_item(item_query_term)
    if exact_item_name.startswith("ERROR:"):
        # If semantic search failed, return a ValidationFailure immediately
        return InventoryValidationFailure(item_name=item_query_term, reason=exact_item_name) # Pass the error message as the reason

    # If successful, proceed with the exact name
    error_item_name = item_query_term # Keep original term for error reporting

    try:
        # --- Step 2: Get Item Details ---
        details_df = get_item_details(exact_item_name)
        if details_df.empty:
            return InventoryValidationFailure(item_name=error_item_name, reason=f"Item '{exact_item_name}' (matched from '{item_query_term}') not found in inventory details.")

        details = details_df.iloc[0]
        unit_price = details["unit_price"]
        min_stock = details["min_stock_level"]

        # --- Step 3: Get Current Stock ---
        stock_df = get_stock_level(exact_item_name, as_of_date)
        stock = stock_df.iloc[0]["current_stock"]

        delivery_note = "In stock."

        # --- Step 4: Handle Out-of-Stock for this Order ---
        if stock < requested_quantity:
            supplier_status = get_supplier_delivery_date(as_of_date, requested_quantity)
            days_to_deliver = (datetime.fromisoformat(supplier_status) - datetime.fromisoformat(as_of_date.split("T")[0])).days

            if days_to_deliver > 7:
                return InventoryValidationFailure(
                    item_name=error_item_name,
                    reason=f"Item '{exact_item_name}' is out of stock ({stock} available) and supplier delivery is too late ({supplier_status})."
                )
            delivery_note = f"Item is out of stock. Supplier will deliver by {supplier_status}."

        # --- Step 5: Handle Auto-Restock ---
        restock_note = f"Stock OK ({stock}/{min_stock})."
        if stock < min_stock:
            reorder_qty = 200
            total_cost = reorder_qty * unit_price
            cash = get_cash_balance(as_of_date)

            if cash >= total_cost:
                create_transaction(
                    item_name=exact_item_name,
                    transaction_type="stock_orders",
                    quantity=reorder_qty,
                    price=total_cost,
                    date=as_of_date
                )
                restock_note = f"Stock was low ({stock}/{min_stock}). Restock order for {reorder_qty} units placed."
            else:
                restock_note = f"Stock is low ({stock}/{min_stock}) but restock failed due to insufficient cash (${cash:.2f} < ${total_cost:.2f})."

        # --- Final Report ---
        return InventoryValidationReport(
            item_name = exact_item_name, # Return the exact name found
            requested_quantity = requested_quantity,
            unit_price = unit_price,
            delivery_note = delivery_note,
            restock_note = restock_note,
            inventory_item_summary=InventoryItemSummary(
                                        item_name = exact_item_name,
                                        stock = stock,
                                        unit_price = unit_price,
                                        value = total_cost
                                    )
            )

    except Exception as e:
         # Log the error for debugging
        print(f"ERROR in tool_validate_item_and_handle_stock for {exact_item_name} (queried as '{item_query_term}'): {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return InventoryValidationFailure(item_name=error_item_name, reason=f"Internal error during validation for '{exact_item_name}': {e}")

# --- Tools for Orchestrator Agent ---
@tool
def tool_generate_financial_report(as_of_date: str) -> FinancialReport:
    """
    Generates a full financial report as a structured Pydantic model.

    Args:
        as_of_date (str): The date for the report (ISO format YYYY-MM-DD).

    Returns:
        FinancialReport: A Pydantic model containing the full report.
    """
    report = generate_financial_report(as_of_date)
    return FinancialReport(
        as_of_date=report["as_of_date"],
        cash_balance=report["cash_balance"],
        inventory_value=report["inventory_value"],
        total_assets=report["total_assets"],
        inventory_summary=[InventoryItemSummary(**item) for item in report["inventory_summary"]],
        top_selling_products=[TopSeller(**item) for item in report["top_selling_products"]]
    )


@tool
def tool_get_full_inventory(as_of_date: str) -> InventoryStockReport:
    """
    Retrieves a list of *all* available items and their stock levels.

    Args:
        as_of_date (str): The date of the request (ISO format YYYY-MM-DD).

    Returns:
        InventoryStockReport: A Pydantic model containing a dictionary of stock levels.
    """
    inventory_dict = get_all_inventory(as_of_date)
    return InventoryStockReport(as_of_date=as_of_date, stock_levels=inventory_dict)


# Set up your agents and create an orchestration agent that will manage them.

# --- Worker Agents ---

inventory_validator = smol.ToolCallingAgent(
    name="inventory_validator",
    description=(
        "You are an expert inventory validator. Your job is to create a Validation Report "
        "by processing each item requested by the user. "
        "You will receive a multi-item order request. You must use the 'as_of_date'. "

        "**Your Workflow:** "
        "1.  **Parse Request:** Identify all item descriptions (query terms) the user asked for and their quantities (e.g., '500 sheets of copy paper', '100 envelopes'). "
        "2.  **Call Processing Tool (Loop):** For *each item query term* and its quantity, you MUST call the `tool_process_order_item` tool. Pass it the `item_query_term`, `requested_quantity`, and `as_of_date`. "
        "    - This single tool handles finding the exact item, validating stock/suppliers, and triggering restocks. It returns a `ValidationSuccess` or `ValidationFailure` model. "
        "3.  **Collect Results:** Gather all the models returned by the tool. "

        "**Final Report:** "
        " - **If ANY `tool_process_order_item` call returned `ValidationFailure`:** "
        "   Respond with 'VALIDATION FAILED:' followed by the `reason` from the first failure model encountered. Make sure to mention the `item_name` (which might be the original query term) from the failure model. "
        " - **If ALL items were processed successfully (returned `ValidationSuccess`):** "
        "   Respond with 'VALIDATION SUCCESS:' followed by the *full string representation* of each `ValidationSuccess` model, each on a new line. "
    ),
    tools=[
        validateInventoryItemTool, # Only tool needed now
    ],
    model=model
)

# Agent 2: (Handles Step 2 - Quoting)
quoting_specialist = smol.ToolCallingAgent(
    name="quoting_specialist",
    description=(  # Correct argument for ToolCallingAgent
        "You are an expert quoting specialist. Your job is to *only* do math and formatting. "
        "You will receive a 'VALIDATION SUCCESS' report from the General Manager. "
        "This report is a string containing one or more `ValidationSuccess` models. "
        "You MUST NOT call any inventory tools. "

        "**Your Workflow:** "
        "1.  **Parse Report:** Parse the string to find all `ValidationSuccess` models. "
        "    From each model, extract `item_name`, `requested_quantity`, `unit_price`, `delivery_note`, and `restock_note`. "
        "2.  **Calculate Price:** Calculate `Total = sum(requested_quantity * unit_price)` for all items. "
        "3.  **Apply Discounts:** Apply 5% discount if Total > $500. Apply 10% discount if Total > $1000. "
        "4.  **Search History:** Call `tool_search_quote_history`. This returns a `QuoteHistoryResponse` model. "
        "5.  **Format Quote:** Respond with 'QUOTE READY:'. Include the itemized list, the final total, any discount, and all 'notes' from the validation models. "
        "    Example: 'QUOTE READY: "
        "    - A4 paper (500 units): $25.00 "
        "    - Envelopes (100 units): $5.00 "
        "    Total: $30.00 (No discount applied). "
        "    Notes: Restock order for A4 paper placed. Envelopes are in stock. "
        "    History: Found 2 similar quotes. "
        "    '"
    ),
    tools=[
        tool_search_quote_history,  # Only tool needed
    ],
    model=model
)

# Agent 3: (Handles Step 3 - Recording)
order_processor = smol.ToolCallingAgent(
    name="order_processor",
    description=(  # Correct argument for ToolCallingAgent
        "You are a meticulous database recorder. Your ONLY job is to record transactions for a sale that is already confirmed. "
        "You will receive the final, approved quote details (item list, quantities, prices, and 'as_of_date'). "
        "1.  **Process Sales (Loop):** For *each item*, call `tool_create_sale_transaction`. "
        "    This tool will return a `TransactionConfirmation` model. "
        "2.  **Final Report:** After processing all items, provide a single summary message: "
        "    'SALE CONFIRMED: All transactions recorded.' and include the details from the `TransactionConfirmation` models."
    ),
    tools=[
        tool_create_sale_transaction,  # Only tool needed
    ],
    model=model
)

# --- Orchestrator Agent ---
@tool
def delegate_to_inventory_validator(request: str) -> str:
    """
    Delegate validation tasks to the inventory validator agent.

    Args:
        request: The full user request to validate

    Returns:
        Validation result message starting with either 'VALIDATION SUCCESS' or 'VALIDATION FAILED'
    """
    return inventory_validator.run(request)


@tool
def interactWithQuotingSpecialist(validation_result: InventoryValidationReport, exact_inventory_item_name : str) -> str:
    """
    Delegate quote generation to the quoting specialist agent.

    Args:
        validation_result: The full inventory validation report from the inventory validator
        exact_inventory_item_name: The exact name of the inventory item that was validated

    Returns:
        Quote message starting with 'QUOTE READY'
    """
    quotingSpecialistQuery= f"""
                            Generating quotation for {exact_inventory_item_name}.
                            Please utilize the inventory validation result for quotation generation: 
                            {validation_result}                            
                            """
    return quoting_specialist.run(quotingSpecialistQuery)


@tool
def delegate_to_order_processor(quote_details: str) -> str:
    """
    Delegate order recording to the order processor agent.

    Args:
        quote_details: The full quote details including items, quantities, prices, and date

    Returns:
        Confirmation message starting with 'SALE CONFIRMED'
    """
    return order_processor.run(quote_details)


# Now create the general manager with the delegation tools
general_manager = smol.ToolCallingAgent(
    name="general_manager",
    description=(
        "You are the General Manager (Orchestrator). You execute a fixed sequential workflow. "
        "You MUST use the 'as_of_date' from the user's request. "

        "**Your primary job is to follow this logic:** "

        "**1. Triage Request:** "
        "   - **Simple Inventory Question?** (e.g., 'Do you have A4 paper?'): "
        "       - Use your own `tool_validate_item_and_handle_stock` (with quantity 1). It returns a `ValidationResult` model. "
        "       - Report the `stock` from the model. Job done. "
        "   - **Report Request?** (e.g., 'Generate report...'): "
        "       - Use your own `tool_generate_financial_report`. It returns a `FinancialReport` model. "
        "       - Present this to the user. Job done. "
        "   - **Order or Quote Request?** (e.g., 'I need...', 'Quote for...'): "
        "       - **This triggers the 3-step sequential workflow.** Proceed to Step 2. "

        "**2. Step 1: Validate (Delegate to inventory_validator)** "
        "   - Use `delegate_to_inventory_validator` with the entire user request. "

        "**3. Analyze Validation Response** "
        "   - **If response starts with 'VALIDATION FAILED':** "
        "       - Relay this failure message to the user. Job done. "
        "   - **If response starts with 'VALIDATION SUCCESS':** "
        "       - The order is valid. You MUST proceed to Step 4. "

        "**4. Step 2: Quote (Delegate to quoting_specialist)** "
        "   - Use `delegate_to_quoting_specialist` with the full 'VALIDATION SUCCESS' string and exact inventory item names returned inventory validator. "

        "**5. Analyze Quote Response** "
        "   - Receive the 'QUOTE READY' response from the quoter. "
        "   - **DO NOT** ask the user for confirmation. The sale is automatic. Proceed to Step 6. "

        "**6. Step 3: Record (Delegate to order_processor)** "
        "   - Use `delegate_to_order_processor` with the quote details. "
        "   - This request MUST include the itemized list, quantities, prices, and `as_of_date` from the quote. "

        "**7. Final Confirmation** "
        "   - Receive the 'SALE CONFIRMED' message from the order processor. "
        "   - Relay this confirmation to the user, combining it with the quote details. "

        "**Rules:** "
        "-   Do NOT reveal internal agent names. "
        "-   Just provide the final, synthesized answer or rejection."
    ),
    tools=[
        tool_generate_financial_report,
        tool_get_full_inventory,
        validateInventoryItemTool,
        delegate_to_inventory_validator,
        interactWithQuotingSpecialist,
        delegate_to_order_processor,
    ],
    model=model
)


# Define the function to be called by the test harness
def call_your_multi_agent_system(request: str) -> str:
    """
    This function is the main entry point for the multi-agent system.
    It passes the user request to the general_manager (orchestrator).

    Args:
        request (str): The full user request, including the date context.

    Returns:
        str: The final, customer-facing response from the agent system.
    """
    print(f"--- AGENT SYSTEM INPUT --- \n{request}\n")
    try:
        # Use smol.act() for a direct request-response with the orchestrator
        response = general_manager.run(request)
        print(f"--- AGENT SYSTEM RESPONSE --- \n{response}\n")
        # Sanitize response to remove sensitive internal info, if any
        if "Error:" in response or "smol.Agent" in response:
            return "I apologize, but I encountered an internal error trying to process your request."
        return response
    except Exception as e:
        print(f"--- AGENT SYSTEM ERROR --- \n{e}\n")
        raise e
        return f"An error occurred while processing your request: {e}"


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    print("Initializing Database...")
    # Initialize with the db_engine and a consistent seed for reproducible results
    init_database(db_engine, seed=137)

    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv").iloc[:1, :]
        # Ensure correct date parsing
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    print("Multi-agent system initialized and ready.")

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")

        pre_report = generate_financial_report(request_date)
        current_cash = pre_report["cash_balance"]
        current_inventory = pre_report["inventory_value"]
        print(f"Cash Balance (pre-request): ${current_cash:.2f}")
        print(f"Inventory Value (pre-request): ${current_inventory:.2f}")

        # Process request
        request_with_date = (
            f"Customer Request: '{row['request']}' "
            f"(Context: Job is '{row['job']}', Event is '{row['event']}'. "
            f"IMPORTANT: You must use this date for all actions: {request_date})"
        )

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = call_your_multi_agent_system(request_with_date)

        # Update state *after* processing the request
        post_report = generate_financial_report(request_date)
        current_cash = post_report["cash_balance"]
        current_inventory = post_report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash (post-request): ${current_cash:.2f}")
        print(f"Updated Inventory (post-request): ${current_inventory:.2f}")

        results.append(
            {
                #"request_id": row['id'],  # Use the original ID from the CSV
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)  # To avoid potential API rate limits

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_local_report_data = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_local_report_data.get('cash_balance', 0):.2f}")
    print(f"Final Inventory: ${final_local_report_data.get('inventory_value', 0):.2f}")
    print("\nTop 5 Selling Products:")
    print(pd.DataFrame(final_local_report_data.get('top_selling_products', [])).to_string())

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    try:
        results = run_test_scenarios()
    except Exception as e:
        #print(f"An error occurred during test run: {e}")
        raise e