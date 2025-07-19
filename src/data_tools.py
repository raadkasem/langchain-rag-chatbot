#!/usr/bin/env python3
"""
Custom tools for querying structured data (CSV and SQLite).
These tools can be used by LangChain agents to answer questions about structured data.
"""

import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain.tools import Tool
from pydantic import BaseModel, Field

class EmployeeSearchInput(BaseModel):
    """Input schema for employee search tool."""
    query: str = Field(..., description="Search query for employee information (name, department, position, etc.)")

class CustomerSearchInput(BaseModel):
    """Input schema for customer search tool."""
    query: str = Field(..., description="Search query for customer information (company, industry, tier, etc.)")

class DataQueryInput(BaseModel):
    """Input schema for general data query tool."""
    query: str = Field(..., description="SQL-like query description or natural language query about the data")

class DataTools:
    """Collection of tools for querying structured data."""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory
        self.db_path = os.path.join(data_directory, "company.db")
        self.employees_csv = os.path.join(data_directory, "employees.csv")
        self.customers_csv = os.path.join(data_directory, "customers.csv")
        
        # Load CSV data into memory for faster access
        self.employees_df = None
        self.customers_df = None
        self._load_csv_data()
    
    def _load_csv_data(self):
        """Load CSV data into pandas DataFrames."""
        try:
            if os.path.exists(self.employees_csv):
                self.employees_df = pd.read_csv(self.employees_csv)
                print(f"✅ Loaded {len(self.employees_df)} employee records")
            
            if os.path.exists(self.customers_csv):
                self.customers_df = pd.read_csv(self.customers_csv)
                print(f"✅ Loaded {len(self.customers_df)} customer records")
                
        except Exception as e:
            print(f"❌ Error loading CSV data: {str(e)}")
    
    def search_employees(self, query: str) -> str:
        """Search for employees based on name, department, position, etc."""
        if self.employees_df is None:
            return "Employee data not available."
        
        try:
            query = query.lower().strip()
            
            # Search across multiple columns
            mask = (
                self.employees_df['name'].str.lower().str.contains(query, na=False) |
                self.employees_df['department'].str.lower().str.contains(query, na=False) |
                self.employees_df['position'].str.lower().str.contains(query, na=False) |
                self.employees_df['email'].str.lower().str.contains(query, na=False)
            )
            
            results = self.employees_df[mask]
            
            if results.empty:
                return f"No employees found matching '{query}'"
            
            # Format results
            output = f"Found {len(results)} employee(s) matching '{query}':\n\n"
            for _, emp in results.iterrows():
                output += f"• {emp['name']} - {emp['position']} in {emp['department']}\n"
                output += f"  Email: {emp['email']}, Hired: {emp['hire_date']}, Salary: ${emp['salary']:,}\n\n"
            
            return output.strip()
            
        except Exception as e:
            return f"Error searching employees: {str(e)}"
    
    def search_customers(self, query: str) -> str:
        """Search for customers based on company name, industry, tier, etc."""
        if self.customers_df is None:
            return "Customer data not available."
        
        try:
            query = query.lower().strip()
            
            # Search across multiple columns
            mask = (
                self.customers_df['company_name'].str.lower().str.contains(query, na=False) |
                self.customers_df['contact_name'].str.lower().str.contains(query, na=False) |
                self.customers_df['industry'].str.lower().str.contains(query, na=False) |
                self.customers_df['subscription_tier'].str.lower().str.contains(query, na=False)
            )
            
            results = self.customers_df[mask]
            
            if results.empty:
                return f"No customers found matching '{query}'"
            
            # Format results
            output = f"Found {len(results)} customer(s) matching '{query}':\n\n"
            for _, customer in results.iterrows():
                output += f"• {customer['company_name']} ({customer['industry']})\n"
                output += f"  Contact: {customer['contact_name']} ({customer['email']})\n"
                output += f"  Tier: {customer['subscription_tier']}, Revenue: ${customer['monthly_revenue']}/month\n\n"
            
            return output.strip()
            
        except Exception as e:
            return f"Error searching customers: {str(e)}"
    
    def get_department_stats(self, query: str = "") -> str:
        """Get statistics about departments."""
        if self.employees_df is None:
            return "Employee data not available."
        
        try:
            stats = self.employees_df.groupby('department').agg({
                'name': 'count',
                'salary': ['mean', 'min', 'max']
            }).round(2)
            
            output = "Department Statistics:\n\n"
            for dept in stats.index:
                count = stats.loc[dept, ('name', 'count')]
                avg_salary = stats.loc[dept, ('salary', 'mean')]
                min_salary = stats.loc[dept, ('salary', 'min')]
                max_salary = stats.loc[dept, ('salary', 'max')]
                
                output += f"• {dept}: {count} employees\n"
                output += f"  Average Salary: ${avg_salary:,.0f}\n"
                output += f"  Salary Range: ${min_salary:,.0f} - ${max_salary:,.0f}\n\n"
            
            return output.strip()
            
        except Exception as e:
            return f"Error getting department stats: {str(e)}"
    
    def get_customer_revenue_stats(self, query: str = "") -> str:
        """Get revenue statistics from customers."""
        if self.customers_df is None:
            return "Customer data not available."
        
        try:
            total_revenue = self.customers_df['monthly_revenue'].sum()
            avg_revenue = self.customers_df['monthly_revenue'].mean()
            
            # Revenue by tier
            tier_stats = self.customers_df.groupby('subscription_tier').agg({
                'monthly_revenue': ['sum', 'count', 'mean']
            }).round(2)
            
            # Revenue by industry
            industry_stats = self.customers_df.groupby('industry').agg({
                'monthly_revenue': ['sum', 'count']
            }).round(2)
            
            output = f"Customer Revenue Statistics:\n\n"
            output += f"Total Monthly Revenue: ${total_revenue:,}\n"
            output += f"Average Revenue per Customer: ${avg_revenue:.2f}\n\n"
            
            output += "Revenue by Subscription Tier:\n"
            for tier in tier_stats.index:
                total = tier_stats.loc[tier, ('monthly_revenue', 'sum')]
                count = tier_stats.loc[tier, ('monthly_revenue', 'count')]
                avg = tier_stats.loc[tier, ('monthly_revenue', 'mean')]
                output += f"• {tier}: ${total:,} total ({count} customers, avg ${avg:.2f})\n"
            
            output += "\nRevenue by Industry:\n"
            industry_revenue = industry_stats.sort_values(('monthly_revenue', 'sum'), ascending=False)
            for industry in industry_revenue.index:
                total = industry_revenue.loc[industry, ('monthly_revenue', 'sum')]
                count = industry_revenue.loc[industry, ('monthly_revenue', 'count')]
                output += f"• {industry}: ${total:,} ({count} customers)\n"
            
            return output
            
        except Exception as e:
            return f"Error getting revenue stats: {str(e)}"
    
    def execute_sql_query(self, query_description: str) -> str:
        """Execute a SQL query based on natural language description."""
        if not os.path.exists(self.db_path):
            return "Database not available."
        
        try:
            # This is a simplified implementation
            # In a production system, you'd want to use an LLM to convert natural language to SQL
            
            conn = sqlite3.connect(self.db_path)
            
            # Simple query mapping based on keywords
            query_lower = query_description.lower()
            
            if "count" in query_lower and "employee" in query_lower:
                query = "SELECT COUNT(*) as employee_count FROM employees"
            elif "count" in query_lower and "customer" in query_lower:
                query = "SELECT COUNT(*) as customer_count FROM customers"
            elif "average salary" in query_lower or "avg salary" in query_lower:
                query = "SELECT AVG(salary) as average_salary FROM employees"
            elif "highest salary" in query_lower or "max salary" in query_lower:
                query = "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1"
            elif "departments" in query_lower and "list" in query_lower:
                query = "SELECT DISTINCT department FROM employees"
            elif "industries" in query_lower and "list" in query_lower:
                query = "SELECT DISTINCT industry FROM customers"
            else:
                return f"I couldn't understand how to query for: '{query_description}'. Try asking about employee counts, salary statistics, or department/industry lists."
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Format output
            if len(df) == 1 and len(df.columns) == 1:
                return f"Result: {df.iloc[0, 0]}"
            else:
                return df.to_string(index=False)
                
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def get_langchain_tools(self) -> List[Tool]:
        """Get LangChain Tool objects for use with agents."""
        
        tools = [
            Tool(
                name="search_employees",
                description="Search for employees by name, department, position, or email. Input should be a search term.",
                func=self.search_employees,
                args_schema=EmployeeSearchInput
            ),
            Tool(
                name="search_customers", 
                description="Search for customers by company name, contact name, industry, or subscription tier. Input should be a search term.",
                func=self.search_customers,
                args_schema=CustomerSearchInput
            ),
            Tool(
                name="get_department_statistics",
                description="Get statistics about employee departments including counts and salary information. No input required.",
                func=self.get_department_stats
            ),
            Tool(
                name="get_revenue_statistics",
                description="Get customer revenue statistics by tier and industry. No input required.",
                func=self.get_customer_revenue_stats
            ),
            Tool(
                name="execute_data_query",
                description="Execute queries about the data like 'count employees', 'average salary', 'list departments', etc. Input should be a natural language description of what you want to query.",
                func=self.execute_sql_query,
                args_schema=DataQueryInput
            )
        ]
        
        return tools

def test_data_tools():
    """Test the data tools functionality."""
    print("🧪 Testing Data Tools")
    print("=" * 50)
    
    # Initialize data tools
    data_tools = DataTools()
    
    # Test employee search
    print("\n👥 Testing Employee Search:")
    print(data_tools.search_employees("engineering"))
    print("\n" + "-" * 30)
    print(data_tools.search_employees("sarah"))
    
    # Test customer search
    print("\n🏢 Testing Customer Search:")
    print(data_tools.search_customers("technology"))
    print("\n" + "-" * 30)
    print(data_tools.search_customers("enterprise"))
    
    # Test statistics
    print("\n📊 Testing Department Statistics:")
    print(data_tools.get_department_stats())
    
    print("\n💰 Testing Revenue Statistics:")
    print(data_tools.get_customer_revenue_stats())
    
    # Test SQL queries
    print("\n🔍 Testing SQL Queries:")
    queries = [
        "count employees",
        "average salary",
        "highest salary",
        "list departments"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print(f"Result: {data_tools.execute_sql_query(query)}")
    
    print("\n✅ Data tools test completed!")

if __name__ == "__main__":
    test_data_tools()