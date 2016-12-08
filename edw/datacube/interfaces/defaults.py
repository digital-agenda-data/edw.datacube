""" Default values for the IDataCubeSettings registry
"""


DATACUBE_THUMBNAIL = u"++resource++scoreboard.theme.images/generic_thumbnail_for_datasets.png"
VISUALIZATION_THUMBNAIL = u"++resource++scoreboard.theme.images/column_chart.png"


DEFAULT_SPARQL_ENDPOINT = u"http://virtuoso.scoreboard.edw.ro/sparql"

DEFAULT_USER_SPARQL_ENDPOINT = u"http://digital-agenda-data.eu/data/sparql"

DEFAULT_CR_URL = u"http://digital-agenda-data.eu/data"

FIND_COUNTRIES_QUERY = \
u"""
# This is a sample SPARQL query that returns the first 100 countries
# included in the data for indicator "%(notation)s" and year "2012"


PREFIX cube: <http://purl.org/linked-data/cube#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dad-prop: <http://semantic.digital-agenda-data.eu/def/property/>

SELECT DISTINCT ?country WHERE {
  ?observation a cube:Observation .
  ?observation dad-prop:indicator <%(indicator)s> .
#  ?observation dad-prop:time-period <http://reference.data.gov.uk/id/gregorian-year/2012> .
  ?observation dad-prop:ref-area [skos:prefLabel ?country]
}
ORDER BY ?country
LIMIT 100
""".strip()


FIND_BREAKDOWNS_QUERY = \
u"""
# This is a sample SPARQL query that returns the first 100 breakdowns
# included in the data for indicator "%(notation)s" and year "2012"


PREFIX cube: <http://purl.org/linked-data/cube#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dad-prop: <http://semantic.digital-agenda-data.eu/def/property/>

SELECT DISTINCT ?breakdown_group ?breakdown (COUNT(?observation) as ?observations) WHERE {
  ?observation a cube:Observation .
  ?observation dad-prop:indicator <%(indicator)s> .
  ?observation dad-prop:breakdown ?breakdown_uri .
  ?breakdown_uri skos:prefLabel ?breakdown .
  ?breakdown_uri dad-prop:membership [
    dad-prop:member-of ?breakdown_group_uri ;
    dad-prop:order ?order
  ] .
  ?breakdown_group_uri skos:prefLabel ?breakdown_group .
  ?breakdown_group_uri dad-prop:order ?group_order.

}
GROUP BY ?breakdown_group ?group_order ?breakdown ?order
ORDER BY ?group_order ?order
LIMIT 1000
""".strip()

