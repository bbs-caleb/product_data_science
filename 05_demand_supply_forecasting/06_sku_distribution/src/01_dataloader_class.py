"""Data Loader Implementation."""
import json


class DataLoader:
    """
    A class to load and parse JSON data for managing forecast demand,
    warehouse stocks, bundle sizes, and transport costs for various products
    and marketplaces.
    """

    def __init__(self, json_file):
        """
        Initializes the DataLoader class with the specified JSON file path.

        Args:
            json_file (str): The path to the JSON file to be loaded.
        """
        self.json_file = json_file
        self.data = self.load_json()

        self.marketplaces = ["Ozz", "WB", "YM"]
        self.bundle_types = ["small", "middle", "large"]

        self.products = []
        self.forecast_demand = {}
        self.warehouse_stock = {}
        self.bundle_sizes = {}
        self.transport_costs = {}

        self.parse_data()

    def load_json(self):
        """
        Loads data from the specified JSON file.

        Returns:
            dict: The data loaded from the JSON file as a dictionary.
                  If the file is not found or there is an error in loading,
                  returns None.
        Raises:
            FileNotFoundError: If the specified file is not found.
            JSONDecodeError: If there is an error in reading the JSON file.
        """
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.json_file}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(
                f"Error decoding JSON from file: {self.json_file}",
                doc="",
                pos=0
            )

    def parse_data(self):
        """
        Parses the loaded data from the JSON file and organizes it into
        forecast demand, warehouse stock, bundle sizes, and transport costs.
        """
        if self.data is None:
            return

        # Check required keys
        required_keys = [
            "forecast_demand",
            "warehouse_stocks",
            "bundle_sizes",
            "transport_costs"
        ]
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"Missing required key: {key}")

        # Parse forecast_demand
        # Format: {"product_sku": {"marketplace": forecast_percentiles}}
        for item in self.data["forecast_demand"]:
            product = item["product"]
            marketplace = item["marketplace"]
            percentiles = item["forecast_percentiles"]

            if product not in self.forecast_demand:
                self.forecast_demand[product] = {}

            self.forecast_demand[product][marketplace] = percentiles

        # Parse warehouse_stocks
        # Format: {"product_sku": quantity}
        for item in self.data["warehouse_stocks"]:
            product = item["product"]
            quantity = item["quantity"]
            self.warehouse_stock[product] = quantity

        # Parse bundle_sizes
        # Format: {"product_sku": {"bundle_type": quantity}}
        for item in self.data["bundle_sizes"]:
            product = item["product"]
            bundle_type = item["type"]
            quantity = item["quantity"]

            if product not in self.bundle_sizes:
                self.bundle_sizes[product] = {}

            self.bundle_sizes[product][bundle_type] = quantity

        # Parse transport_costs
        # Format: {"product_sku": {"marketplace": {"bundle_type": cost}}}
        for item in self.data["transport_costs"]:
            for product, marketplaces_data in item.items():
                if product not in self.transport_costs:
                    self.transport_costs[product] = {}

                for marketplace, costs in marketplaces_data.items():
                    self.transport_costs[product][marketplace] = costs

        # Extract unique products list
        all_products = set()
        all_products.update(self.forecast_demand.keys())
        all_products.update(self.warehouse_stock.keys())
        all_products.update(self.bundle_sizes.keys())
        all_products.update(self.transport_costs.keys())

        self.products = sorted(list(all_products))
