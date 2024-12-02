using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Truck : MonoBehaviour
{
    public float speed = 1f;         // Forward movement speed
    public float turnSpeed = 50f;    // Turning speed
    public Animator armAnimator;      // Reference to the Animator
    public Animator handleAnimator;      // Reference to the Animator
    private bool _isAnimating = false;  // Prevent input while animating

    void Update()
    {
        // Move the car forward
        float move = speed * Time.deltaTime;
        transform.Translate(Vector3.back * move);

        // Get player input for steering
        float steerInput = Input.GetAxis("Horizontal"); // A/D or Left/Right Arrow Keys

        // Rotate the car
        if (steerInput != 0)
        {
            float turn = steerInput * turnSpeed * Time.deltaTime;
            transform.Rotate(0, turn, 0);
        }
        
        if (Input.GetKeyDown(KeyCode.Space) && !_isAnimating)
        {
            armAnimator.SetTrigger("ArmMove");
            handleAnimator.SetTrigger("HandleMove");
            _isAnimating = true;
        }
        // Check if the wave animation is finished by looking at the current animator state
        AnimatorStateInfo stateInfo = armAnimator.GetCurrentAnimatorStateInfo(0);

        // If the wave animation has finished, allow input again
        if (stateInfo.IsName("Arm") && stateInfo.normalizedTime >= 1.0f)
        {
            armAnimator.Play("Idle");
            handleAnimator.Play("Idle");

            _isAnimating = false;
        }
    }

}
