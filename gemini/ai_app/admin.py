from django.contrib import admin
from.models import AIResponse

#Zmiana pola z obrazem jako tylko do odczytu
class AIResponseAdmin(admin.ModelAdmin):
    readonly_fields = ('response','title', 'content')

admin.site.register(AIResponse, AIResponseAdmin, editable=True)