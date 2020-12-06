# OpenRefine

### Background
OpenRefine is a data wrangling tool that runs in your browser that can perform bulk functions on datasets to "clean" them up. It's like a lovechild of Excel and databases. It was developed by a company called Metaweb under the name Freebase Gridworks. Metaweb was bought out by Google, who renamed the tool Google Refine. In turn Google decided it would rather not support this tool and since 2012 it has become open source. 

### Datasets
I've queried the Storm Events Database from NOAA's National Centers for Environmental Information for the following information:
* Search Results for Allegheny County, Pennsylvania
* Event Types: Excessive Heat, Extreme Cold/Wind Chill, Flash Flood, Flood, Heat, Heavy Rain, Heavy Snow, High Wind, Lake-Effect Snow, Strong Wind, Thunderstorm Wind, Winter Storm
* Allegheny county contains the following zones:
'Allegheny'
* 437 events were reported between 01/01/2015 and 05/01/2020 (1948 days)
* Column Definitions: 'Mag': Magnitude, 'Dth': Deaths, 'Inj': Injuries, 'PrD': Property Damage, 'CrD': Crop Damage
* Wind Magnitude Definitions: Measured Gust:'MG', Estimated Gust:'EG', Measured Sustained:'MS', Estimated Sustained:'ES'

You can import just about any dataset into OpenRefine and use the tool to wrangle your data.

Steps to start a project:
* Find a dataset
* Upload a dataset 
* Configure parsing options
    * CSV/TSV
    * JSON
    * etc.
    * Name & Tag


### Facets
Facets describe the data and offer us filtering capabilites. OpenRefine treats data much in the same way that pandas does by using the concepts of index and series. You can think of all the columns as series identified individually by their index (or row). When we transform a series we're changing each index in the series. 

Let's look at some facets:
* CZ_NAME_STR
    * This series refers to the county zone name.
* BEGIN_DATE
    * The date the severe weather event started.
* BEGIN_TIME
    * The time of day the severe weather event started
* EVENT_TYPE
    * Which type of the pre-defined weather events occurred.

#### Text Facets, Filtering, and Editing
Text facets are strings and can be manipulated. Click the box to the left of the header, select `Facet`, then `Text Facet` which will open up a window on the left side task bar. Here we can select one of the two options to filter the data, either by 'ALLEGHENY (ZONE)' or 'ALLEGHENY CO.'. You can click on either of the two facet values to filter out any value that does not match the clicked on value.

The same think happens if we select the facet view for EVENT TYPE. Notice how if you select one in either box the other facet's count changed dynamically.

You can filter the data very selectively too. If you hover over a value in either selected facet, and click `include` that value is included in your filter (i.e. all other values are not shown). You can include or exclude as many as you'd like.

Another filtering tool which the facets provide is `Facet by choice counts` which is found below the last value in your facet list. If you click this you will see a histogram with a numeric representation of the count of your facet values. You can slide each of the poles dynamically across the range and if a facet value falls within that range it is filtered in the project. If I slide the maximum pole (on the right) from 190 down to 50 we see that all events ocurring more than 50 times are filtered out.

Let's say we want to edit the facets. Maybe we have multiple categories that are similar or we don't like the way the dataset initially displayed them. Click the `edit` button to the right of the facet value and a small dialog box will appear. Here you can change the string to what you'd like it to be. For example I'm changing ALLEGHENY CO. to Allegheny_County and ALLEGHENY (ZONE) to Allegheny_Zone. This will change all series values that match ALLEGHENY CO. to Allegheny_County. Same thing with the zone facet.

Similarly editing the values to the same can also concatenate them. Let's combined High Wind and Strong Winds in our EVENT_TYPES. Here we just change the facet value of High Wind to Strong Wind and the indeces in the series are changed and combined.

Another useful feature here is the `Cluster` tool. If you click the cluster button towards the top of the page you will open a dialog box. This tool finds similar facet values and clusters them together. Here you can make bulk edits to normalize the data, because it is often the case that database do not contain a standard entry format. Let's select merge ALLEGHENY CO. and allegheny_co. and select a new value for the output rows. We'll do the same for the second cluster.

#### Numeric Facets, Common Transformations, and Tabulations
Severe weather often comes with extensive damage to property. Let's take a look to see what happened in our dataset. Let's open the DAMAGE_PROPERTY_NUM facet by selecting `Facet` then `Numeric Facet`. But what happens is that no numeric values are indicated even though we know for a fact that the dataset has a wide range of property damage amounts in USD. Let's see what happens if we open it as a text facet.
All the numbers are now appearing, but it looks like they're being interpreted as a string. That makes it hard to do any tabulations or aggregations since you can't do that with strings. 
What we need to do here is perform a transformation, turning one datatype into another. This is very much like assigning a datatype in Python. To do this click on the dropdown menu button to the left of DAMAGE_PROPERTY_NUM and hover over `Edit` then we want to hover over `Common Transformations` (these are preset transformations for common data tranformations), finally we want to select `To number`, which will convert all string representations of the numbers in DAMAGE_PROPERTY_NUM from a string to a number. When it's finished, notice how the cell values are right-aligned and green. This is another indicator that our values are now numbers.
Now we can select the numeric facet for DAMAGE_PROPERTY_NUM and we'll see a sliding scale appear in our facet toolbar on the left which we can use as a filter. For individual sums of property damage however we can still use the text facet to view the values. Let's do the same transformation to DAMAGE_CROPS_NUM so we can sum each to get a idea of the total monetary impact for a storm considering both crops and property.
To do that we need to create a new column. Click the dropdown from DAMAGE_CROPS_NUM and hover over `Edit Column` > `Add column based on this column`. Here a new dialog box will open and we'll get the option to name the columns header. Then we want to enter the code that's going to perform our summation:
* `grel:cells["DAMAGE_PROPERTY_NUM"].value + cells["DAMAGE_CROPS_NUM"].value return value` 
OpenRefine uses both GREL and Python/Jython as languages to perform different operations. I'm not 100% sure what the syntax is but the code I used to combine the two is written in GREL. The nice feature here is that we'll get a preview of what'll happen. When you're done click OK and your new column will be entered into the dataset.

#### Timeline Facets
Timelines facets deal with exactly what they sound like: date values. Let's open the timeline facet for BEGIN_DATE. We see here that it's not formatted correctly. If we open it up the series as a text facet we can see the range of dates in the dataset. Let's conduct a transformation to change the datatype. If we open the timeline facet for BEGIN_DATE again we can see the distribution in a histogram. We can use the poles on either side to filter down to a specific date range. Notice also that it provides a nice visualization of when severe weather has occurred.


### Undo/Redo and Exports
Let's say you've made a mistake or need to revert back to an earlier state of the dataset. Control/Command + Z isn't going to work here. But! We can see each and every step we took. At the top of the Facet/Filter toolbar we can switch over to the Undo/Redo tab and see the edit history of the dataset. If you click on any of the previous steps the dataset will undo the n-latest steps and revert to the state it was at the step you selected.
Let's say that you downloaded another dataset from NOAA's NCEI with similar values for another county with the same exact query. You don't want to have to do all that work again right? Fret not, because we can extract all the work we've done as a json file. At the top of the facet toolbar click on the Undo/Redo tab again and then click `Extract...` . In the dialog box that appears you can select individual edits or all of them. Simply copy the json in the right hand side box and save it in your preferred text editor. You can open up your new data set, click on the Undo/Redo tab again and click `Apply...`. In the dialog box that appears this time, paste the json you saved from the extract stage and click `Perform Operations`. Presto, all the work you've done will be applied to the new dataset.
Once your data is cleaned you can export it a project (in tar.gz form) which can be re-uploaded to OpenRefine when you need it. Otherwise you can export the dataset as a csv or tsv.


### Homework
If you'd like download OpenRefine and try your hand at some data wrangling. It doesn't have to be fancy, just try the tool out and see how you like it.


**Fair warning**: OpenRefine struggles with huge datasets (such as the PennDOT crash dataset (1997-2018) Kayla mentioned). Try something smaller first because you'll either crash the program or your machine will self-immolate. 

## Sources: 

### Installation Instructions
* [From the official Github repo](https://github.com/OpenRefine/OpenRefine/wiki/Installation-Instructions)

### Usage
* [University of Illinois OpenRefine Guide](https://guides.library.illinois.edu/openrefine)
* [OpenRefine screencasts](https://github.com/OpenRefine/OpenRefine/wiki/Screencasts)
* [GREL Functions](https://github.com/OpenRefine/OpenRefine/wiki/GREL-Functions)
* [Cleaning Data with OpenRefine by John Little](https://libjohn.github.io/openrefine/index.html)


### Data
* [My severe weather dataset](https://www.ncdc.noaa.gov/stormevents/listevents.jsp?eventType=%28Z%29+Excessive+Heat&eventType=%28Z%29+Extreme+Cold%2FWind+Chill&eventType=%28C%29+Flash+Flood&eventType=%28Z%29+Flood&eventType=%28Z%29+Heat&eventType=%28C%29+Heavy+Rain&eventType=%28Z%29+Heavy+Snow&eventType=%28Z%29+High+Wind&eventType=%28Z%29+Lake-Effect+Snow&eventType=%28Z%29+Strong+Wind&eventType=%28C%29+Thunderstorm+Wind&eventType=%28Z%29+Winter+Storm&beginDate_mm=01&beginDate_dd=01&beginDate_yyyy=2015&endDate_mm=05&endDate_dd=01&endDate_yyyy=2020&county=ALLEGHENY%3A3&hailfilter=0.00&tornfilter=0&windfilter=000&sort=DT&submitbutton=Search&statefips=42%2CPENNSYLVANIA)
    * [Data dictionary](https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/Storm-Data-Export-Format.pdf)
* [Wester PA Regional Data Center](http://www.wprdc.org/)
* [NOAA's NCEI database](https://www.ncdc.noaa.gov/stormevents/)






