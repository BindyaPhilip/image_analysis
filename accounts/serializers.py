# Import Django REST Framework's serializers module to convert models to JSON and validate API input
from rest_framework import serializers
# Import models from the current app to define their serialization behavior
from .models import User, FarmerProfile, ExpertProfile, CropType, AvailabilitySlot, ConsultationBooking, CommunityPost, Feedback, SystemMetric,PasswordResetToken
from django.contrib.auth.hashers import make_password
# Serializer for CropType model to handle crop type data (e.g., "Maize")
# class CropTypeSerializer(serializers.ModelSerializer):
#     class Meta:
#         # Specify the model to serialize
#         model = CropType
#         # Fields to include in the JSON output/input validation
#         fields = ['id', 'name']

# Serializer for User model to handle user data (e.g., email, role)
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'username', 'password', 'role', 'phone_number', 'is_approved']
        extra_kwargs = {
            'password': {'write_only': True},
            'id': {'read_only': True},
            'role': {'read_only':True},
            'is_approved':{'read_only':True}
        }

    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data['password']) # type: ignore
        validated_data['is_active'] = True
        return super().create(validated_data)
    
    def update(self, instance, validated_data):
        instance.email = validated_data.get('email',instance.email)
        instance.username = validated_data.get('username', instance.username)
        instance.phone_number = validated_data.get('phone_number', instance.phone_number)
        instance.save()
        return instance

# Serializer for FarmerProfile model to handle farmer-specific data
class FarmerProfileSerializer(serializers.ModelSerializer):
    # Nested serializer for crop_types (many-to-many relationship), allowing full crop type details
    crop_types = serializers.ListField(
        child=serializers.ChoiceField(choices=CropType.choices()),
        allow_empty=True,
        required=False
    )
    # Nested serializer for user, read-only to prevent modifying user data via profile updates
    user = UserSerializer(read_only=True)

    class Meta:
        # Specify the model to serialize
        model = FarmerProfile
        # Fields to include, covering all farmer profile attributes
        fields = ['id', 'user', 'farm_location', 'farm_size', 'crop_types', 'soil_type', 'irrigation_method', 'disease_history', 'farm_latitude', 'farm_longitude', 'experience_years', 'preferred_language', 'farm_equipment']

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret['crop_types'] = instance.crop_types.split(',') if instance.crop_types else []
        return ret

    def to_internal_value(self, data):
        ret = super().to_internal_value(data)
        crop_types = data.get('crop_types', [])
        if crop_types:
            valid_choices = [choice[0] for choice in CropType.choices()]
            for ct in crop_types:
                if ct not in valid_choices:
                    raise serializers.ValidationError(f"Invalid crop type: {ct}")
            ret['crop_types'] = ','.join(crop_types)
        else:
            ret['crop_types'] = ''
        return ret
    
class RegisterFarmerSerializer(serializers.Serializer):
    email = serializers.EmailField()
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, style={'input_type': 'password'})
    confirm_password = serializers.CharField(write_only=True, style={'input_type': 'password'})
    phone_number = serializers.CharField(max_length=15, required=False, allow_blank=True)
    farm_location = serializers.CharField(max_length=200)
    farm_size = serializers.FloatField()
    crop_types = serializers.ListField(
        child=serializers.ChoiceField(choices=CropType.choices()),
        allow_empty=True,
        required=False
    )

    def validate(self, data):
        if data['password'] != data['confirm_password']:
            raise serializers.ValidationError({"confirm_password": "Passwords do not match"})
        return data

    def create(self, validated_data):
        validated_data.pop('confirm_password')
        crop_types = validated_data.pop('crop_types', [])
        user_data = {
            'email': validated_data['email'],
            'username': validated_data['username'],
            'phone_number': validated_data.get('phone_number', ''),
            'role': 'farmer',
            'is_active': True  # Ensure user is active
        }
        user = User.objects.create_user(**user_data, password=validated_data['password'])
        profile_data = {
            'user': user,
            'farm_location': validated_data['farm_location'],
            'farm_size': validated_data['farm_size'],
            'crop_types': ','.join(crop_types) if crop_types else ''
        }
        profile = FarmerProfile.objects.create(**profile_data)
        return {'user': UserSerializer(user).data, 'profile': FarmerProfileSerializer(profile).data}


# Serializer for ExpertProfile model to handle expert-specific data
class ExpertProfileSerializer(serializers.ModelSerializer):
    # Nested serializer for user, read-only to prevent modifying user data
    user = UserSerializer(read_only=False, partial = True)

    class Meta:
        # Specify the model to serialize
        model = ExpertProfile
        # Fields to include, covering all expert profile attributes
        fields = ['id', 'user', 'areas_of_expertise', 'certifications', 'bio', 'experience_years', 'institution', 'languages_spoken', 'social_links']
    def update(self, instance, validated_data):
        # Handle nested user data
        user_data = validated_data.pop('user', None)
        if user_data:
            user_serializer = UserSerializer(instance.user, data=user_data, partial=True)
            if user_serializer.is_valid():
                user_serializer.save()
            else:
                raise serializers.ValidationError(user_serializer.errors)

        # Update ExpertProfile fields
        return super().update(instance, validated_data)

# Serializer for AvailabilitySlot model to handle expert consultation slots
class AvailabilitySlotSerializer(serializers.ModelSerializer):
    # Nested serializer for expert, read-only to show expert details without modification
    expert = UserSerializer(read_only=True)

    class Meta:
        # Specify the model to serialize
        model = AvailabilitySlot
        # Fields to include, covering slot details
        fields = ['id', 'expert', 'start_time', 'end_time', 'is_booked']

# Serializer for ConsultationBooking model to handle farmer-expert bookings
class ConsultationBookingSerializer(serializers.ModelSerializer):
    # Nested serializer for farmer, read-only to show farmer details
    farmer = UserSerializer(read_only=True)
    # Nested serializer for slot, read-only to show slot details
    slot = AvailabilitySlotSerializer(read_only=True)

    class Meta:
        # Specify the model to serialize
        model = ConsultationBooking
        # Fields to include, covering booking details
        fields = ['id', 'farmer', 'slot', 'created_at']

# Serializer for CommunityPost model to handle farmer posts
class CommunityPostSerializer(serializers.ModelSerializer):
    # Nested serializer for farmer, read-only to show post author
    farmer = UserSerializer(read_only=True)

    class Meta:
        # Specify the model to serialize
        model = CommunityPost
        # Fields to include, covering post details
        fields = ['id', 'farmer', 'title', 'content', 'created_at', 'is_approved', 'is_flagged']

# Serializer for Feedback model to handle farmer feedback
class FeedbackSerializer(serializers.ModelSerializer):
    # Nested serializer for farmer, read-only to show feedback author
    farmer = UserSerializer(read_only=True)

    class Meta:
        # Specify the model to serialize
        model = Feedback
        # Fields to include, covering feedback details
        fields = ['id', 'farmer', 'content', 'created_at']

# Serializer for SystemMetric model to handle system performance metrics
class SystemMetricSerializer(serializers.ModelSerializer):
    class Meta:
        # Specify the model to serialize
        model = SystemMetric
        # Fields to include, covering metric details
        fields = ['id', 'metric_name', 'value', 'recorded_at']

class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True, style={'input_type': 'password'})
    new_password = serializers.CharField(write_only=True, style={'input_type': 'password'})
    confirm_password = serializers.CharField(write_only=True, style={'input_type': 'password'})

    def validate(self, data):
        if data['new_password'] != data['confirm_password']:
            raise serializers.ValidationError({"confirm_password": "New passwords do not match"})
        user = self.context['request'].user
        if not user.check_password(data['old_password']):
            raise serializers.ValidationError({"old_password": "Incorrect old password"})
        return data
    
class ForgotPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()

    def validate_email(self, value):
        try:
            User.objects.get(email=value)
        except User.DoesNotExist:
            raise serializers.ValidationError("No user found with this email")
        return value
    

class ResetPasswordSerializer(serializers.Serializer):
    token = serializers.UUIDField()
    new_password = serializers.CharField(write_only=True, style={'input_type': 'password'})
    confirm_password = serializers.CharField(write_only=True, style={'input_type': 'password'})

    def validate(self, data):
        if data['new_password'] != data['confirm_password']:
            raise serializers.ValidationError({"confirm_password": "Passwords do not match"})
        try:
            reset_token = PasswordResetToken.objects.get(token=data['token'])
            if not reset_token.is_valid():
                raise serializers.ValidationError({"token": "Token has expired"})
        except PasswordResetToken.DoesNotExist:
            raise serializers.ValidationError({"token": "Invalid token"})
        return data