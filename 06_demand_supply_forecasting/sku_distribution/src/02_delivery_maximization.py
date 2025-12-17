"""Data Loader and Delivery Maximization Implementation."""
import json
import pulp


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

        required_keys = [
            "forecast_demand",
            "warehouse_stocks",
            "bundle_sizes",
            "transport_costs"
        ]
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"Missing required key: {key}")

        for item in self.data["forecast_demand"]:
            product = item["product"]
            marketplace = item["marketplace"]
            percentiles = item["forecast_percentiles"]

            if product not in self.forecast_demand:
                self.forecast_demand[product] = {}

            self.forecast_demand[product][marketplace] = percentiles

        for item in self.data["warehouse_stocks"]:
            product = item["product"]
            quantity = item["quantity"]
            self.warehouse_stock[product] = quantity

        for item in self.data["bundle_sizes"]:
            product = item["product"]
            bundle_type = item["type"]
            quantity = item["quantity"]

            if product not in self.bundle_sizes:
                self.bundle_sizes[product] = {}

            self.bundle_sizes[product][bundle_type] = quantity

        for item in self.data["transport_costs"]:
            for product, marketplaces_data in item.items():
                if product not in self.transport_costs:
                    self.transport_costs[product] = {}

                for marketplace, costs in marketplaces_data.items():
                    self.transport_costs[product][marketplace] = costs

        all_products = set()
        all_products.update(self.forecast_demand.keys())
        all_products.update(self.warehouse_stock.keys())
        all_products.update(self.bundle_sizes.keys())
        all_products.update(self.transport_costs.keys())

        self.products = sorted(list(all_products))


class DeliveryMaximization:
    """
    A class to model and solve a linear programming problem for maximizing
    deliveries of products to multiple marketplaces.
    """

    def __init__(
        self,
        products,
        marketplaces,
        forecast_demand,
        warehouse_stock,
        bundle_sizes,
        transport_costs,
        min_percentile,
        max_percentile,
    ):
        """
        Initializes the DeliveryMaximization class with necessary data.
        """
        self.products = products
        self.marketplaces = marketplaces
        self.forecast_demand = forecast_demand
        self.warehouse_stock = warehouse_stock
        self.bundle_sizes = bundle_sizes
        self.transport_costs = transport_costs

        self._valid_percentiles = [10, 25, 50, 75, 90]

        if min_percentile not in self._valid_percentiles:
            raise ValueError(
                f"min_percentile must be one of {self._valid_percentiles}"
            )

        if max_percentile not in self._valid_percentiles:
            raise ValueError(
                f"max_percentile must be one of {self._valid_percentiles}"
            )

        if min_percentile > max_percentile:
            raise ValueError("min_percentile cannot be greater than max_percentile")

        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

        self.model = pulp.LpProblem("Maximize_Delivery", pulp.LpMaximize)
        self.decision_vars = {}

    def _define_decision_variables(self):
        """
        Defines the decision variables for each product, marketplace,
        and bundle size.

        Variable Naming Format:
            x_{product}_{marketplace}_{bundle_size}
        """
        bundle_types = ["small", "middle", "large"]

        for product in self.products:
            for marketplace in self.marketplaces:
                for bundle_type in bundle_types:
                    var_name = f"x_{product}_{marketplace}_{bundle_type}"
                    self.decision_vars[var_name] = pulp.LpVariable(
                        var_name,
                        lowBound=0,
                        cat="Integer"
                    )

    def _define_objective(self):
        """
        Defines the objective function for the model.

        Objective Function:
            Maximize: Σ (bundle_sizes[p][b] * x_{p}_{m}_{b})
        """
        bundle_types = ["small", "middle", "large"]
        objective_terms = []

        for product in self.products:
            for marketplace in self.marketplaces:
                for bundle_type in bundle_types:
                    var_name = f"x_{product}_{marketplace}_{bundle_type}"
                    bundle_size = self.bundle_sizes[product][bundle_type]
                    objective_terms.append(
                        bundle_size * self.decision_vars[var_name]
                    )

        self.model += pulp.lpSum(objective_terms), "Total_Delivery"

    def _add_constraints(self):
        """
        Adds the necessary constraints to the model.

        Constraints:
            1. Warehouse stock limits
            2. Maximum delivery (max_percentile)
            3. Minimum delivery (min_percentile)
        """
        bundle_types = ["small", "middle", "large"]

        # 1. Warehouse Stock Constraint for each product
        for product in self.products:
            stock_terms = []
            for marketplace in self.marketplaces:
                for bundle_type in bundle_types:
                    var_name = f"x_{product}_{marketplace}_{bundle_type}"
                    bundle_size = self.bundle_sizes[product][bundle_type]
                    stock_terms.append(
                        bundle_size * self.decision_vars[var_name]
                    )

            constraint_name = f"Warehouse_Stock_{product}"
            self.model += (
                pulp.lpSum(stock_terms) <= self.warehouse_stock[product],
                constraint_name
            )

        # 2. Max Delivery Constraint for each product and marketplace
        for product in self.products:
            for marketplace in self.marketplaces:
                delivery_terms = []
                for bundle_type in bundle_types:
                    var_name = f"x_{product}_{marketplace}_{bundle_type}"
                    bundle_size = self.bundle_sizes[product][bundle_type]
                    delivery_terms.append(
                        bundle_size * self.decision_vars[var_name]
                    )

                max_demand = self.forecast_demand[product][marketplace][
                    str(self.max_percentile)
                ]
                constraint_name = f"MaxDelivery_{product}_{marketplace}"
                self.model += (
                    pulp.lpSum(delivery_terms) <= max_demand,
                    constraint_name
                )

        # 3. Min Delivery Constraint for each product and marketplace
        for product in self.products:
            for marketplace in self.marketplaces:
                delivery_terms = []
                for bundle_type in bundle_types:
                    var_name = f"x_{product}_{marketplace}_{bundle_type}"
                    bundle_size = self.bundle_sizes[product][bundle_type]
                    delivery_terms.append(
                        bundle_size * self.decision_vars[var_name]
                    )

                min_demand = self.forecast_demand[product][marketplace][
                    str(self.min_percentile)
                ]
                constraint_name = f"MinDelivery_{product}_{marketplace}"
                self.model += (
                    pulp.lpSum(delivery_terms) >= min_demand,
                    constraint_name
                )

    def solve(self):
        """
        Solves the linear programming model.
        """
        self._define_decision_variables()
        self._define_objective()
        self._add_constraints()

        self.model.solve(pulp.PULP_CBC_CMD(msg=0))

    def get_results(self):
        """
        Retrieves the results of the optimization.

        Returns:
            dict: A dictionary where the key is (product, marketplace)
                  and value is total delivered quantity.
        """
        bundle_types = ["small", "middle", "large"]
        results = {}

        for product in self.products:
            for marketplace in self.marketplaces:
                total_delivered = 0.0
                for bundle_type in bundle_types:
                    var_name = f"x_{product}_{marketplace}_{bundle_type}"
                    var_value = pulp.value(self.decision_vars[var_name]) or 0
                    bundle_size = self.bundle_sizes[product][bundle_type]
                    total_delivered += var_value * bundle_size

                results[(product, marketplace)] = total_delivered

        return results
