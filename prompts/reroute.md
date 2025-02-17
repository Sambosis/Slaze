Your job is to create custom routing program for specific business needs. There is a csv in your project directory named reroute.csv
It has the following columms:
RT,DAY,STOP,NEW_RT,NEW_DAY,NEW_STOP,CUST_NUM,CUST_NAME,STREET_NUMBER,ADDRESS,CITY,ZIP,CHEM_SALES,TOTAL_SALES,LATITUDE,LONGITUDE
Here are what they mean:
RT: This is the Route number. In this business a route takes 19 days to run. The driver of a route is responsible for running stops on each day of their assigned route.
DAY: This is a number which represents which out of the 19 days of the route this customer is scheduled on.
STOP: This is a number which represents the order in which the customer will be visited but the route driver their scheduled route day.
NEW_RT,NEW_DAY,NEW_STOP: will respectively be the new values after the re-routing is completed.
CUST_NUM,CUST_NAME,STREET_NUMBER,ADDRESS,CITY,ZIP are all info about the customer.
CHEM_SALES: refer to the dollar amount of sales the customer has on average of chemicals.
TOTAL_SALES: refer to the dollar amount of total sales the customer has on average of all products, including chemicals.
LATITUDE,LONGITUDE are the LATITUDE,LONGITUDE coordinates of the customer's location.
The customers start and end each day from a depot located at 8912 Yellow Brick rd. Rosedale, MD 21237  Latitude = 39.341270 LONGITUDE = -76.479261.
The routing proogram should reasign the customers to new routes and new days and new stops.  The goal is to minimize the total distance traveled by the drivers.

One of the routes should be bigger than the other 2 (there will be 3 routes in total.)  That largest route should have approximately 40% of the total chemical sales and approximately 40% of the total.  The other 2 should each have the remaining 30% of the 2 sales numbers.
You need to use sophisticated methods to optimize the routes and will still need to combine several different approaches to get the best results.
After writing the optimzation program, save the new RT, DAY AND STOP TO THE NEW_ versions of the files.
The customers on each RT should all be in the same general area as much as possible. 
There should be a map created showing all of their stops. with lines drawing a line from the first stop of the day and the last stop of the day.
There is no need to draw a line from the depot and the last stop nor should there be a line from the depot to the first stop of the day.  
Identify  each RT with a unique color.
You should provide a before and after map as well as an updated csv.
display the open the the maps up  to check for accuracy.