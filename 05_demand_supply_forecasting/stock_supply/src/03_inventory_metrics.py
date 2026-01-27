"""Inventory Metrics Implementation."""


class InventoryMetrics:
    """A class to calculate inventory management metrics."""

    def __init__(self, products):
        """
        Initializes the InventoryMetrics object with a list of products.

        Args:
            products (list of dict): List of dictionaries, each representing a product.
            Each dictionary should contain:
                - 'name' (str): Name of the product.
                - 'price' (float): Selling price per unit.
                - 'cost' (float): Cost per unit.
                - 'quantity_sold' (int): Number of units sold.
                - 'inventory_start' (float): Value of inventory at the start of the period.
                - 'inventory_end' (float): Value of inventory at the end of the period.
        """
        self.products = products

    def gmv(self):
        """Calculates the total Gross Merchandise Value (GMV).

        Returns:
            float: The total GMV, which is the sum of the product of price
                   and quantity sold for each product.
        """
        total_gmv = sum(
            product["price"] * product["quantity_sold"]
            for product in self.products
        )
        return round(total_gmv, 2)

    def gross_margin(self):
        """Calculates the total Gross Margin.

        Returns:
            tuple: A tuple containing:
                - margin (float): The total Gross Margin.
                - margin_percentage (float): The Gross Margin percentage.
        """
        total_revenue = sum(
            product["price"] * product["quantity_sold"]
            for product in self.products
        )
        total_cogs = sum(
            product["cost"] * product["quantity_sold"]
            for product in self.products
        )

        margin = total_revenue - total_cogs

        if total_revenue == 0:
            margin_percentage = 0
        else:
            margin_percentage = (margin / total_revenue) * 100

        return round(margin, 2), round(margin_percentage, 2)

    def average_inventory_cost(self):
        """Calculates the total Average Inventory Cost.

        Returns:
            float: The total Average Inventory Cost.
        """
        total_avg_inventory = sum(
            (product["inventory_start"] + product["inventory_end"]) / 2
            for product in self.products
        )
        return round(total_avg_inventory, 2)

    def gmroi(self):
        """Calculates the total Gross Margin Return on Investment (GMROI).

        Returns:
            float: The GMROI, calculated as gross margin divided by
                   average inventory cost.
        """
        margin, _ = self.gross_margin()
        avg_inventory = self.average_inventory_cost()

        if avg_inventory == 0:
            return 0

        gmroi_value = margin / avg_inventory
        return round(gmroi_value, 2)

    def inventory_turnover(self):
        """Calculates the total Inventory Turnover.

        Returns:
            float: The Inventory Turnover, calculated as total cost of goods
                   sold divided by average inventory cost.
        """
        total_cogs = sum(
            product["cost"] * product["quantity_sold"]
            for product in self.products
        )
        avg_inventory = self.average_inventory_cost()

        if avg_inventory == 0:
            return 0

        turnover = total_cogs / avg_inventory
        return round(turnover, 2)

    def turnover_period(self):
        """Calculates the total Turnover Period (in days).

        Returns:
            float: The Turnover Period, calculated as 365 divided by
                   Inventory Turnover.
        """
        turnover = self.inventory_turnover()

        if turnover == 0:
            return 0

        period = 365 / turnover
        return round(period, 2)
