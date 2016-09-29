""" DataCube vocabulary
"""
from zope.interface import implements
from zope.schema.interfaces import IVocabularyFactory
from zope.schema.vocabulary import SimpleVocabulary
from zope.schema.vocabulary import SimpleTerm
from Products.CMFCore.utils import getToolByName

class DataCubeVocabulary(object):
    """ All DataCube object
    """
    implements(IVocabularyFactory)

    def __call__(self, context):
        ctool = getToolByName(context, 'portal_catalog')
        if not ctool:
            return SimpleVocabulary([])

        brains = ctool(portal_type='DataCube')
        items = [SimpleTerm('', '', 'No cloning')]
        for brain in brains:
            items.append(SimpleTerm(brain.UID, brain.UID, brain.Title))
        return SimpleVocabulary(items)

class RelatedVisualizationVocabulary(object):
    """ All DataCube object
    """
    implements(IVocabularyFactory)

    def __call__(self, context):
        items = [SimpleTerm('', '', 'None')]

        for relation in context.getBRefs():
            items.append(SimpleTerm(relation.UID(), relation.UID(), relation.Title()))
        return SimpleVocabulary(items)
