"""Definition of the DataCube content type
"""

from zope.interface import implements

from Products.Archetypes import atapi
from Products.ATContentTypes.content import folder
from Products.ATContentTypes.content import schemata

from edw.datacube.interfaces import IDataCube
from edw.datacube.config import PROJECTNAME, _
from edw.datacube.data.cube import Cube


DataCubeSchema = folder.ATFolderSchema.copy() + atapi.Schema((
    atapi.TextField(
        'endpoint',
        schemata="default",
        default="http://virtuoso.scoreboard.edw.ro/sparql",
        required=True,
        default_content_type='text/plain',
        allowable_content_types=('text/plain',),
        languageIndependent=True,
        widget=atapi.StringWidget(
            label=_(u"SPARQL Endpoint"),
        )
    ),
    atapi.StringField(
        'extended_title',
        schemata="default",
        widget=atapi.StringWidget(
            label=_(u"Title"),
        )
    ),
    atapi.TextField(
        'summary',
        required=False,
        searchable=True,
        validators=('isTidyHtmlWithCleanup',),
        default_output_type='text/x-html-safe',
        widget=atapi.RichWidget(
            label=_(u"Description"),
            allow_file_upload=False)
        ),
    atapi.TextField(
        'dataset',
        schemata="default",
        required=False,
        default_content_type='text/plain',
        allowable_content_types=('text/plain',),
        languageIndependent=True,
        widget=atapi.TextAreaWidget(
            label=_(u"Dataset"),
        )
    ),
    atapi.ImageField(
        'thumbnail',
        schemata="default",
        required=False,
        languageIndependent=True,
        widget=atapi.ImageWidget(
            label=_(u"Thumbnail"),
        )
    ),
))

# Set storage on fields copied from ATFolderSchema, making sure
# they work well with the python bridge properties.

DataCubeSchema['title'].storage = atapi.AnnotationStorage()
DataCubeSchema['title'].widget.label = _(u'Identifier')
DataCubeSchema['description'].storage = atapi.AnnotationStorage()
DataCubeSchema['description'].widget.visible = {'view': 'invisible',
                                                'edit': 'invisible'}
DataCubeSchema['dataset'].widget.visible = {'view': 'invisible',
                                            'edit': 'hidden'}
DataCubeSchema.moveField('endpoint', before='title')
DataCubeSchema.moveField('extended_title', before='title')

for field in DataCubeSchema.values():
    if field.schemata != 'default':
        field.schemata = 'default'
        field.widget.visible = {'view': 'invisible', 'edit': 'invisible'}

schemata.finalizeATCTSchema(
    DataCubeSchema,
    folderish=False,
    moveDiscussion=False
)

class DataCube(folder.ATFolder):
    """Description of the Example Type"""
    implements(IDataCube)

    meta_type = "DataCube"
    schema = DataCubeSchema

    title = atapi.ATFieldProperty('title')
    description = atapi.ATFieldProperty('description')

    # -*- Your ATSchema to Python Property Bridges Here ... -*-

    def get_cube(self, endpoint=''):
        endpoint = endpoint or self.getEndpoint()
        return Cube(endpoint, self.getDataset())

    def displayContentsTab(self):
        return False

atapi.registerType(DataCube, PROJECTNAME)
